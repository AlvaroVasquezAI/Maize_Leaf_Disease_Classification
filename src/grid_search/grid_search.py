import itertools
import pandas as pd
import torch
import os
from datetime import datetime
from ..models.vit_model import MaizeViTModel
from ..training.trainer import Trainer
from sklearn.metrics import classification_report
import json
import numpy as np
from tqdm import tqdm

class GridSearchCV:
    def __init__(self, param_grid, train_loader, val_loader, test_loader, model_name='google/vit-base-patch16-224'):
        self.param_grid = param_grid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories for results
        self.results_dir = os.path.join('models', 'grid_search', 'results')
        self.models_dir = os.path.join('models', 'grid_search', 'best_model')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        self.best_model = None
        self.best_params = None
        self.best_val_score = 0.0
        
    def evaluate_model(self, model, data_loader):
        """Evaluate model on given data loader"""
        model.eval()
        all_preds = []
        all_labels = []
        running_loss = 0.0
        
        # For per-class accuracy
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                running_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                
                # Get predictions and labels
                preds = predicted.cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                # Calculate per-class accuracy
                for pred, label in zip(preds, labels):
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate standard metrics
        report = classification_report(
            all_labels,
            all_preds,
            target_names=data_loader.dataset.get_class_names(),
            output_dict=True
        )
        
        # Add per-class accuracy
        class_names = data_loader.dataset.get_class_names()
        for class_idx, class_name in enumerate(class_names):
            if class_idx in class_total:
                accuracy = class_correct[class_idx] / class_total[class_idx]
                report[class_name]['accuracy'] = accuracy * 100
        
        # Add average loss
        report['loss'] = running_loss / len(data_loader)
        
        # Print detailed report
        print("\nDetailed Classification Report:")
        print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Accuracy':>10} {'Support':>10}")
        print("-" * 75)
        
        for class_name in class_names:
            metrics = report[class_name]
            print(f"{class_name:<25} "
                f"{metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} "
                f"{metrics['f1-score']:>10.4f} "
                f"{metrics.get('accuracy', 0):>10.1f}% "
                f"{metrics['support']:>10}")
        
        print("-" * 75)
        print(f"{'Accuracy':<25} {'':<10} {'':<10} {'':<10} "
            f"{report['accuracy']*100:>10.1f}% "
            f"{sum(class_total.values()):>10}")
        
        return report
        
    def run(self):
        # Generate all parameter combinations
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in itertools.product(*self.param_grid.values())]
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Add progress bar for combinations
        for i, params in enumerate(tqdm(param_combinations, desc="Grid Search Progress")):
            print(f"\nCombination {i+1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            try:
                # Initialize model with current parameters
                model = MaizeViTModel(
                    num_classes=6,
                    pretrained_model=self.model_name,
                    hidden_dropout_prob=params['hidden_dropout_prob'],
                    attention_probs_dropout_prob=params['attention_probs_dropout_prob']
                )
                model.to(self.device)
                
                # Initialize trainer
                trainer_config = {
                    'learning_rate': params['learning_rate'],
                    'weight_decay': params['weight_decay'],
                    'batch_size': params['batch_size'],
                    'scheduler_patience': params['scheduler_patience'],
                    'scheduler_factor': params['scheduler_factor']
                }
                
                trainer = Trainer(
                    model=model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    config=trainer_config,
                    device=self.device
                )
                
                # Train model
                print(f"Training with parameters: {params}")
                trainer.train(params['num_epochs'])
                
                # Evaluate on all sets
                train_metrics = self.evaluate_model(model, self.train_loader)
                val_metrics = self.evaluate_model(model, self.val_loader)
                test_metrics = self.evaluate_model(model, self.test_loader)
                
                # Store results
                # In the run method, when storing results:
                result = {
                    'params': params,
                    'train_accuracy': train_metrics['accuracy'] * 100,  # Convert to percentage
                    'train_loss': train_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'] * 100,      # Convert to percentage
                    'val_loss': val_metrics['loss'],
                    'test_accuracy': test_metrics['accuracy'] * 100,    # Convert to percentage
                    'test_loss': test_metrics['loss'],
                    'timestamp': timestamp,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
                self.results.append(result)
                
                # Update best model if necessary
                if val_metrics['accuracy'] > self.best_val_score:
                    self.best_val_score = val_metrics['accuracy']
                    self.best_params = params
                    
                    # Save best model
                    model_save_path = os.path.join(self.models_dir, f'best_model_{timestamp}.pth')
                    torch.save({
                        'state_dict': model.vit.state_dict(),  # Save only the ViT state dict
                        'params': params,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'test_metrics': test_metrics,
                        'timestamp': timestamp
                    }, model_save_path)
                    
                    print(f"New best model saved with validation accuracy: {val_metrics['accuracy']:.2f}")
                
                trainer.clear_memory()  # Clear gradients
                del trainer           # Delete trainer instance
                del model            # Delete model
                torch.cuda.empty_cache()  # Final cache clear

            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                # Cleanup in case of error
                if 'trainer' in locals(): 
                    trainer.clear_memory()
                    del trainer
                if 'model' in locals(): 
                    del model
                torch.cuda.empty_cache()
                continue
            
            # Save results after each iteration
            self.save_results(timestamp)
    
    def save_results(self, timestamp):
        try:
            # Create results directory for this run
            run_dir = os.path.join(self.results_dir, timestamp)
            os.makedirs(run_dir, exist_ok=True)
            
            # Save detailed results
            results_df = pd.DataFrame([
                {
                    **result['params'],
                    'train_accuracy': result['train_accuracy'],
                    'train_loss': result['train_loss'],
                    'val_accuracy': result['val_accuracy'],
                    'val_loss': result['val_loss'],
                    'test_accuracy': result['test_accuracy'],
                    'test_loss': result['test_loss']
                }
                for result in self.results
            ])
            
            # Save results DataFrame
            results_path = os.path.join(run_dir, 'grid_search_results.csv')
            results_df.to_csv(results_path, index=False)
            print(f"Saved results to: {results_path}")
            
            # Save detailed metrics for each run
            for i, result in enumerate(self.results):
                metrics_path = os.path.join(run_dir, f'run_{i}_metrics.json')
                metrics = {
                    'parameters': result['params'],
                    'train_metrics': result['train_metrics'],
                    'val_metrics': result['val_metrics'],
                    'test_metrics': result['test_metrics']
                }
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                
            # Save best parameters
            if self.best_params:
                best_params_path = os.path.join(run_dir, 'best_params.json')
                with open(best_params_path, 'w') as f:
                    json.dump(self.best_params, f, indent=4)
                print(f"Saved best parameters to: {best_params_path}")
                
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def get_total_combinations(self):
        total = 1
        for values in self.param_grid.values():
            total *= len(values)
        return total

    def estimate_total_time(self, time_per_epoch=10):
        total_combinations = self.get_total_combinations()
        total_epochs = sum([epochs for epochs in self.param_grid['num_epochs']])
        estimated_hours = (total_combinations * total_epochs * time_per_epoch) / 3600
        return estimated_hours