import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from src.models.vit_model import MaizeViTModel
from src.data.dataset import MaizeLeafDataset

def create_arg_parser():
    """Creates and returns the ArgumentParser object."""
    parser = argparse.ArgumentParser(description="Evaluate the best Maize Leaf Disease classification model.")
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/grid_search/best_model/best_model_20250227_133601.pth',
        help='Path to the trained model checkpoint (.pth file), relative to the project root.'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'validation'],
        help='The dataset split to evaluate on: "test" or "validation".'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference.'
    )
    return parser

def load_model_for_inference(model_path, num_classes, device):
    """Loads the custom MaizeViTModel and prepares it for inference."""
    print(f"Loading model from: {model_path}")
    model = MaizeViTModel(num_classes=num_classes, pretrained_model='google/vit-base-patch16-224')

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.vit.load_state_dict(checkpoint['state_dict'])
            print("Successfully loaded model from a grid search checkpoint.")
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded model from a standard training checkpoint.")
        else:
            model.vit.load_state_dict(checkpoint)
            print("Successfully loaded model from a raw state_dict file.")
    except Exception as e:
        print(f"Error loading the model checkpoint: {e}")
        exit()

    model.to(device)
    model.eval()
    print("Model is loaded and in evaluation mode.")
    return model

def run_inference(model, data_loader, device):
    """Runs inference on the provided data_loader and returns predictions and true labels."""
    model.eval()
    all_predictions, all_true_labels = [], []
    print(f"Running inference on the {data_loader.dataset.split_name} set...")
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=pixel_values)
            _, predicted_indices = torch.max(outputs.logits, 1)
            all_predictions.extend(predicted_indices.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    return np.array(all_predictions), np.array(all_true_labels)

def generate_evaluation_artifacts(y_true, y_pred, class_names, output_dir, dataset_name, suffix):
    """Generates and saves the classification report and confusion matrix."""
    print("\n" + "="*50)
    print(f"      Classification Report ({dataset_name} Set)")
    print("="*50)
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    report_path = os.path.join(output_dir, f"classification_report_{suffix}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report for the Best Model on the {dataset_name} Set\n\n")
        f.write(report)
    print(f"\nClassification report saved to: {report_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix on {dataset_name} Set', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    matrix_path = os.path.join(output_dir, f"confusion_matrix_{suffix}.png")
    plt.savefig(matrix_path)
    print(f"Confusion matrix plot saved to: {matrix_path}")

def main():
    """Main function to orchestrate the evaluation process."""
    args = create_arg_parser().parse_args()

    # Define output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # === DYNAMIC PATHS AND FILENAMES BASED ON --split ARGUMENT ===
    if args.split == 'validation':
        csv_path = 'data/validation/validation.csv'
        output_suffix = 'validation'
        dataset_name = 'Validation'
    else:  # Default to 'test'
        csv_path = 'data/test/test.csv'
        output_suffix = 'test'
        dataset_name = 'Test'
    
    print(f"--- Starting evaluation on the {dataset_name} set ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model_for_inference(args.model_path, num_classes=6, device=device)

    print(f"\nPreparing {dataset_name} dataset...")
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Use the dynamically selected CSV path
    dataset = MaizeLeafDataset(csv_file=csv_path, feature_extractor=image_processor, train=False)
    # Add a property to the dataset for logging purposes
    dataset.split_name = dataset_name
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(os.cpu_count(), 4))
    print(f"{dataset_name} dataset loaded with {len(dataset)} samples.")

    y_pred, y_true = run_inference(model, data_loader, device)

    class_names = dataset.get_class_names()

    generate_evaluation_artifacts(y_true, y_pred, class_names, output_dir, dataset_name, output_suffix)
    
    print(f"\nEvaluation on {dataset_name} set finished successfully.")

if __name__ == '__main__':
    main()