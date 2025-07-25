import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import os
from sklearn.utils.class_weight import compute_class_weight

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion=None,
        optimizer=None,
        scheduler=None,
        device=None,
        config=None
    ):
        # Initialize basic configurations
        self.config = config if config else {}
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and move to device
        self.model = model
        self.model.to(self.device)
        
        # Initialize data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_to_idx = train_loader.dataset.class_to_idx
        
        # Compute class weights
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['labels'].cpu().numpy())
        
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels
        )
        
        # Increase weights for problematic classes
        southern_rust_idx = self.class_to_idx['Southern Rust']
        phaeosphaeria_idx = self.class_to_idx['Phaeosphaeria Leaf Spot']
        class_weights[southern_rust_idx] *= 2.0
        class_weights[phaeosphaeria_idx] *= 1.5
        
        # Move class weights to device
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Initialize criterion with class weights
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss(weight=class_weights)
        self.criterion.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optimizer if optimizer else optim.AdamW(
            model.parameters(),
            lr=float(self.config.get('learning_rate', 2e-5)),
            weight_decay=float(self.config.get('weight_decay', 0.01))
        )
        
        # Initialize scheduler
        self.scheduler = scheduler if scheduler else ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=int(self.config.get('scheduler_patience', 3)),
            factor=float(self.config.get('scheduler_factor', 0.1))
        )
        
        # Initialize early stopping parameters
        self.patience = self.config.get('early_stopping_patience', 5)
        self.min_delta = self.config.get('early_stopping_delta', 0.001)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        self.scaler = torch.amp.GradScaler()
        
        # Print initialization info
        print(f"Initialized trainer with device: {self.device}")
        if torch.cuda.is_available():
            print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
    def train_epoch(self):
        try:
            """Train for one epoch with mixup and automatic mixed precision"""
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc='Training')
            for batch in pbar:
                try:
                    # Move batch to device with non_blocking=True
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    
                    # Use automatic mixed precision
                    with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        # Apply mixup
                        mixed_inputs, targets_a, targets_b, lam = self.mixup_data(
                            batch['pixel_values'], 
                            batch['labels']
                        )
                        
                        # Forward pass with mixed inputs
                        self.optimizer.zero_grad(set_to_none=True)
                        outputs = self.model(pixel_values=mixed_inputs)
                        
                        # Compute mixed loss
                        loss = self.mixup_criterion(
                            self.criterion,
                            outputs.logits,
                            targets_a,
                            targets_b,
                            lam
                        )
                    
                    # Backward pass with scaled gradients
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = outputs.logits.max(1)
                    total += batch['labels'].size(0)
                    correct += (lam * predicted.eq(targets_a).float() + 
                            (1 - lam) * predicted.eq(targets_b).float()).sum().item()
                    
                    # Update progress bar with GPU memory info
                    gpu_mem = f"{torch.cuda.memory_allocated()/1024**2:.0f}MB" if torch.cuda.is_available() else "N/A"
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{100.*correct/total:.2f}%",
                        'gpu_mem': gpu_mem
                    })
                    
                except Exception as e:
                    print(f"Error in batch: {str(e)}")
                    self.clear_memory()
                    continue
                
                # Optional: Clear memory periodically
                if total % (self.config.get('memory_clear_frequency', 1000)) == 0:
                    self.clear_memory()
            
            return running_loss/len(self.train_loader), 100.*correct/total
            
        finally:
            self.clear_memory()

    def validate(self):
        try:
            """Validate the model"""
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc='Validating')
                for batch in pbar:
                    try:
                        # Move batch to device with non_blocking=True
                        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                        
                        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                            outputs = self.model(**batch)
                            loss = outputs.loss
                        
                        running_loss += loss.item()
                        _, predicted = outputs.logits.max(1)
                        total += batch['labels'].size(0)
                        correct += predicted.eq(batch['labels']).sum().item()
                        
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(batch['labels'].cpu().numpy())
                        
                        # Update progress bar with GPU info
                        gpu_mem = f"{torch.cuda.memory_allocated()/1024**2:.0f}MB" if torch.cuda.is_available() else "N/A"
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'acc': f"{100.*correct/total:.2f}%",
                            'gpu_mem': gpu_mem
                        })
                        
                    except Exception as e:
                        print(f"Error in validation batch: {str(e)}")
                        self.clear_memory()
                        continue
            
            # Calculate metrics
            val_loss = running_loss/len(self.val_loader)
            val_acc = 100.*correct/total
            
            # Get detailed classification report
            report = classification_report(
                all_labels,
                all_preds,
                target_names=self.val_loader.dataset.get_class_names(),
                digits=4
            )
            
            return val_loss, val_acc, report
            
        finally:
            self.clear_memory()

    def train(self, num_epochs):
        try:
            """Full training loop with early stopping"""
            best_val_acc = 0.0
            checkpoint_dir = self.config.get('checkpoint_dir', 'models/checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            
            # Print initial GPU memory status
            if torch.cuda.is_available():
                print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            
            for epoch in range(num_epochs):
                try:
                    print(f'\nEpoch {epoch+1}/{num_epochs}')
                    
                    # Add gradual unfreezing
                    self.model.gradual_unfreeze(epoch)
                    
                    # Training phase
                    train_loss, train_acc = self.train_epoch()
                    
                    # Validation phase
                    val_loss, val_acc, report = self.validate()
                    
                    # Print metrics
                    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
                    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
                    print('\nClassification Report:')
                    print(report)
                    
                    # Print GPU memory status
                    if torch.cuda.is_available():
                        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                    
                    # Early stopping check
                    if val_loss < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_loss
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        
                    if self.early_stopping_counter >= self.patience:
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        break
                    
                    # Update scheduler
                    if self.scheduler is not None:
                        self.scheduler.step(val_loss)
                    
                    # Save best model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                            'scaler_state_dict': self.scaler.state_dict(),
                            'val_acc': val_acc,
                            'val_loss': val_loss,
                            'train_acc': train_acc,
                            'train_loss': train_loss
                        }
                        torch.save(checkpoint, best_model_path)
                        print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

                    # Log learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f'Current learning rate: {current_lr:.2e}')
                    
                    # Clear memory after each epoch
                    self.clear_memory()
                    
                except Exception as e:
                    print(f"Error in epoch {epoch+1}: {str(e)}")
                    self.clear_memory()
                    continue
                    
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise e
            
        finally:
            self.clear_memory()
            if torch.cuda.is_available():
                print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

    def mixup_data(self, x, y, alpha=None):
        """Performs mixup on the input and target
        Args:
            x: input tensor
            y: target tensor
            alpha: mixup alpha value. If None, reads from config
        """
        # Get alpha from config if not provided
        if alpha is None:
            alpha = self.config.get('mixup_alpha', 0.2)  # Default to 0.2 if not in config
        
        # Rest of your code remains the same
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def clear_memory(self):
        """Enhanced memory clearing"""
        try:
            # Clear gradients
            if hasattr(self, 'optimizer'):
                self.optimizer.zero_grad(set_to_none=True)
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                # Force garbage collection before clearing cache
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Optional: Print memory stats
                if self.config.get('debug_memory', False):
                    print(f"GPU Memory after clearing:")
                    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                    print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        
        except Exception as e:
            print(f"Error during memory clearing: {str(e)}")