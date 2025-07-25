from flask import Flask, request, jsonify
import sys
import os
import torch
import time 
from PIL import Image
from flask_cors import CORS
from transformers import ViTImageProcessor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.models.vit_model import MaizeViTModel

app = Flask(__name__)
CORS(app) 

MODEL_PATH = r"C:\Users\alvar\GitHub\MaizeClassificactionDiseases\models\grid_search\best_model\best_model_20250227_133601.pth"
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp") 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_NAMES = [
    'Common Rust', 
    'Gray Leaf Spot', 
    'Healthy', 
    'Northern Leaf Blight',
    'Phaeosphaeria Leaf Spot', 
    'Southern Rust'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
image_processor = None

def load_model():
    global model, image_processor
    
    print("Initializing model...")
    print(f"Using device: {device}")
    
    model = MaizeViTModel(num_classes=len(CLASS_NAMES), pretrained_model='google/vit-base-patch16-224')
    
    print(f"Loading model checkpoint from: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        if hasattr(model, 'vit'): 
            model.vit.load_state_dict(checkpoint['state_dict'])
            print(f"Model weights (model.vit.state_dict) loaded successfully from checkpoint (grid search format).")
        else:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Model weights (model.state_dict) loaded successfully from checkpoint (grid search format, assumed full model).")

    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights (model_state_dict) loaded successfully from checkpoint (standard trainer format).")
    else:
        model.load_state_dict(checkpoint)
        print(f"Model weights loaded successfully (assumed checkpoint IS the state_dict).")
    
    model.to(device) 
    model.eval()  
    
    print("Initializing image processor...")
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    print("Image processor initialized successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    global model, image_processor 
    if model is None or image_processor is None:
        try:
            print("Model or processor not loaded. Attempting to load...")
            load_model()
        except Exception as e:
            print(f"Critical error loading model/processor: {str(e)}")
            return jsonify({'error': f"Error loading model/processor: {str(e)}"}), 500 
 
    if 'image' not in request.files:
        print("No image file found in request.")
        return jsonify({'error': 'No image file provided in the request'}), 400 

    request_processing_start_time = time.time() 
    
    try:
        image_file = request.files['image'] 
        
        temp_image_path = os.path.join(UPLOAD_FOLDER, f"temp_image_{int(time.time())}_{image_file.filename}")
        
        save_file_start_time = time.time()
        image_file.save(temp_image_path)
        save_file_duration_ms = (time.time() - save_file_start_time) * 1000
        print(f"Image saved temporarily to: {temp_image_path} (File save time: {save_file_duration_ms:.2f} ms)")
        
        pil_load_start_time = time.time()
        pil_image = Image.open(temp_image_path).convert('RGB')
        pil_load_duration_ms = (time.time() - pil_load_start_time) * 1000
        print(f"Image loaded with PIL, size: {pil_image.size} (PIL load time: {pil_load_duration_ms:.2f} ms)")
        
        preprocess_start_time = time.time()
        inputs = image_processor(images=pil_image, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        preprocess_duration_ms = (time.time() - preprocess_start_time) * 1000
        print(f"Image preprocessed for ViT (Preprocessing time: {preprocess_duration_ms:.2f} ms)")
        
        model_inference_start_time = time.time()
        if torch.cuda.is_available(): 
            torch.cuda.synchronize(device=device)

        with torch.no_grad():
            outputs = model(**inputs)

        if torch.cuda.is_available(): #
            torch.cuda.synchronize(device=device)
        model_inference_duration_ms = (time.time() - model_inference_start_time) * 1000
        
        postprocess_start_time = time.time()
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1) 
        confidence, predicted_class_index = torch.max(probabilities, dim=1) 
        
        predicted_class_name = CLASS_NAMES[predicted_class_index.item()]
        confidence_value = confidence.item()
        postprocess_duration_ms = (time.time() - postprocess_start_time) * 1000
        
        print(f"Model Inference time: {model_inference_duration_ms:.2f} ms")
        print(f"Output Post-processing time: {postprocess_duration_ms:.2f} ms)")
        print(f"Prediction: Class='{predicted_class_name}', Index={predicted_class_index.item()}, Confidence={confidence_value:.4f}")
        
        try:
            os.remove(temp_image_path)
            print(f"Temporary image {temp_image_path} removed.")
        except Exception as e_remove:
            print(f"Warning: Could not remove temporary image {temp_image_path}: {e_remove}")
        
        total_request_processing_duration_ms = (time.time() - request_processing_start_time) * 1000
        print(f"Total request processing time on server: {total_request_processing_duration_ms:.2f} ms")
 
        return jsonify({
            'className': predicted_class_name,
            'confidence': float(confidence_value),
            'processingTime': float(total_request_processing_duration_ms / 1000.0),
        })
        
    except Exception as e:
        print(f"Error during image prediction: {str(e)}")
        import traceback
        traceback.print_exc() 
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"Temporary image {temp_image_path} removed after error.")
            except Exception as e_remove_err:
                print(f"Warning: Could not remove temporary image {temp_image_path} after error: {e_remove_err}")
        return jsonify({'error': f"An error occurred processing the image: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'UP',
        'model_loaded': model is not None,
        'image_processor_loaded': image_processor is not None,
        'device_in_use': str(device)
    })

if __name__ == '__main__':
    try:
        load_model()
        print(f"Model and processor loaded. Starting Flask server on http://0.0.0.0:5000...")
        app.run(host='0.0.0.0', port=5000, debug=True) 
    except Exception as e:
        print(f"Failed to initialize and start server: {str(e)}")
        import traceback
        traceback.print_exc()