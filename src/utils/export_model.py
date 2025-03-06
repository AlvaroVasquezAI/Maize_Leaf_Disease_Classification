import torch
from src.models.vit_model import MaizeViTModel
from transformers import ViTFeatureExtractor
import logging
import warnings
from torch.jit import TracerWarning

def export_model(checkpoint_path, export_path):
    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=TracerWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # 1. Load the original model
        logger.info("Loading original model...")
        model = MaizeViTModel(num_classes=6)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.vit.load_state_dict(checkpoint['state_dict'])
        else:
            model.vit.load_state_dict(checkpoint)
        
        model.eval()
        
        # 2. Create simplified wrapper
        class MobileViTWrapper(torch.nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.vit_model = vit_model
                
            def forward(self, pixel_values):
                # Simple forward pass without any conditional logic
                outputs = self.vit_model(pixel_values=pixel_values)
                return outputs.logits
        
        # 3. Create wrapper
        logger.info("Creating mobile wrapper...")
        mobile_model = MobileViTWrapper(model.vit)
        mobile_model.eval()
        
        # 4. Verify wrapper
        example_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            original_output = model(pixel_values=example_input)
            wrapper_output = mobile_model(example_input)
            
            diff = (original_output.logits - wrapper_output).abs().max().item()
            logger.info(f"Max difference between original and wrapper: {diff}")
            
            if diff > 1e-6:
                raise ValueError("Wrapper model outputs don't match original model!")
        
        # 5. Trace the model with multiple example inputs
        logger.info("Tracing model...")
        traced_model = torch.jit.trace(
            mobile_model,
            example_input,
            check_trace=True,
            strict=False
        )
        
        # 6. Test traced model
        with torch.no_grad():
            traced_output = traced_model(example_input)
            trace_diff = (original_output.logits - traced_output).abs().max().item()
            logger.info(f"Max difference between original and traced: {trace_diff}")
            
            if trace_diff > 1e-6:
                raise ValueError("Traced model outputs don't match original model!")
        
        # 7. Save model
        logger.info(f"Saving traced model to: {export_path}")
        traced_model.save(export_path)
        
        # 8. Final verification with real image
        logger.info("\nTesting with real image...")
        image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        test_image_path = 'data/test/Gray_Leaf_Spot/1_1.jpg'
        
        from PIL import Image
        test_image = Image.open(test_image_path).convert('RGB')
        inputs = image_processor(images=test_image, return_tensors="pt")
        
        # Load saved model and test
        loaded_model = torch.jit.load(export_path)
        loaded_model.eval()
        
        with torch.no_grad():
            # Original model prediction
            original_pred = model(pixel_values=inputs['pixel_values'])
            original_probs = torch.nn.functional.softmax(original_pred.logits, dim=1)
            
            # Exported model prediction
            exported_pred = loaded_model(inputs['pixel_values'])
            exported_probs = torch.nn.functional.softmax(exported_pred, dim=1)
            
            # Compare predictions
            logger.info("\nPrediction comparison:")
            logger.info(f"Original model prediction: {torch.argmax(original_probs).item()}")
            logger.info(f"Exported model prediction: {torch.argmax(exported_probs).item()}")
            
            prob_diff = (original_probs - exported_probs).abs().max().item()
            logger.info(f"Maximum probability difference: {prob_diff:.6f}")
            
            if prob_diff > 1e-6:
                logger.warning("Warning: Significant difference in predictions!")
            else:
                logger.info("Predictions match exactly!")
        
        logger.info("Model export successful!")
        return True
        
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise e

if __name__ == "__main__":
    CHECKPOINT_PATH = "models/grid_search/best_model/best_model_20250227_133601.pth"
    EXPORT_PATH = "models/mobile_version/model_android.pt"
    
    import os
    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    
    success = export_model(CHECKPOINT_PATH, EXPORT_PATH)
    
    if success:
        print("\nModel export completed successfully!")
        print(f"Exported model size: {os.path.getsize(EXPORT_PATH) / (1024*1024):.2f} MB")