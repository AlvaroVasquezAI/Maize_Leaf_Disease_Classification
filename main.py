import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from src.data.dataset import MaizeLeafDataset
from src.models.vit_model import MaizeViTModel
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, load_config
import os
import argparse

def setup_data_and_model(config):
    """Setup datasets, dataloaders, and model"""
    # Initialize image processor
    print("Initializing ViT image processor...")
    image_processor = ViTImageProcessor.from_pretrained(
        config['model']['name']
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MaizeLeafDataset(
        csv_file=config['data']['train_csv'],
        feature_extractor=image_processor,
        train=True
    )
    
    val_dataset = MaizeLeafDataset(
        csv_file=config['data']['validation_csv'],
        feature_extractor=image_processor,
        train=False
    )
    
    test_dataset = MaizeLeafDataset(
        csv_file=config['data']['test_csv'],
        feature_extractor=image_processor,
        train=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Calculate optimal number of workers
    num_workers = min(os.cpu_count(), config['data_loading'].get('num_workers', 4))
    
    # Create data loaders with optimized settings
    print(f"Creating data loaders with {num_workers} workers...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['data_loading']['pin_memory'],
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['data_loading']['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['data_loading']['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader

def train_model(config, train_loader, val_loader):
    """Regular training process"""
    # Initialize model
    print("Initializing model...")
    model = MaizeViTModel(
        num_classes=config['model']['num_classes'],
        pretrained_model=config['model']['name']
    )
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device
    )
    
    # Create checkpoint directory
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Train model
    print("Starting training...")
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("Training completed!")

def main():
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load configuration
    config_path = os.path.join(base_dir, 'configs', 'config.yaml')
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'grid_search'], 
                       default='train', help='Training mode')
    args = parser.parse_args()
    
    try:
        # Setup data loaders
        train_loader, val_loader, test_loader = setup_data_and_model(config)
        
        if args.mode == 'grid_search':
            print("Starting Grid Search...")
            from src.grid_search.run_grid_search import run_grid_search
            best_model, best_params = run_grid_search(train_loader, val_loader, test_loader)
            print(f"Best parameters found: {best_params}")
        else:
            train_model(config, train_loader, val_loader)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()