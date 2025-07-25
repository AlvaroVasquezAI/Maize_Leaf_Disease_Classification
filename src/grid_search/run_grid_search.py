from .grid_search import GridSearchCV

def run_grid_search(train_loader, val_loader, test_loader):
    # Define parameter grid
    param_grid = {
        'learning_rate': [1e-5, 5e-5],
        'weight_decay': [0.01, 0.02],
        'batch_size': [16, 32],
        'num_epochs': [15],
        'scheduler_patience': [3],
        'scheduler_factor': [0.1],
        'hidden_dropout_prob': [0.1],
        'attention_probs_dropout_prob': [0.1]
    }
    
    # Initialize and run grid search
    grid_search = GridSearchCV(
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

     # Print estimated time
    estimated_hours = grid_search.estimate_total_time()
    print(f"Estimated total time: {estimated_hours:.1f} hours")
    
    # Get user confirmation
    response = input("Do you want to proceed? (y/n): ")
    if response.lower() != 'y':
        return
    
    # Run grid search
    grid_search.run()
    
    return grid_search.best_model, grid_search.best_params