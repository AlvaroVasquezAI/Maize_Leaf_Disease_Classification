import sys
import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from transformers import ViTImageProcessor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.models.vit_model import MaizeViTModel


class MaizeClassifierGUI:
    def __init__(self):
        # Setup window
        self.root = ctk.CTk()
        self.root.title("Maize Leaf Disease Classifier")
        self.root.geometry("800x700")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize model variables
        self.model = None
        self.image_processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names
        self.class_names = [
            'Common Rust',
            'Gray Leaf Spot',
            'Healthy',
            'Northern Leaf Blight',
            'Phaeosphaeria Leaf Spot',
            'Southern Rust'
        ]
        
        self.create_gui()
        
    def create_gui(self):
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Maize Leaf Disease Classifier",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=10)
        
        # Model Selection Section
        self.model_frame = ctk.CTkFrame(self.main_frame)
        self.model_frame.pack(pady=10, fill="x", padx=20)
        
        self.model_label = ctk.CTkLabel(
            self.model_frame,
            text="Step 1: Select Model File (.pth)",
            font=ctk.CTkFont(size=16)
        )
        self.model_label.pack(pady=5)
        
        self.model_button = ctk.CTkButton(
            self.model_frame,
            text="Select Model",
            command=self.load_model_file
        )
        self.model_button.pack(pady=5)
        
        self.model_status = ctk.CTkLabel(
            self.model_frame,
            text="No model loaded",
            font=ctk.CTkFont(size=14)
        )
        self.model_status.pack(pady=5)
        
        # Image Selection Section
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.pack(pady=10, fill="x", padx=20)
        
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="Step 2: Select Image to Classify",
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(pady=5)
        
        self.image_button = ctk.CTkButton(
            self.image_frame,
            text="Select Image",
            command=self.load_image,
            state="disabled"
        )
        self.image_button.pack(pady=5)
        
        # Image display
        self.display_label = ctk.CTkLabel(self.main_frame, text="")
        self.display_label.pack(pady=10)
        
        # Prediction Section
        self.prediction_frame = ctk.CTkFrame(self.main_frame)
        self.prediction_frame.pack(pady=10, fill="x", padx=20)
        
        self.prediction_label = ctk.CTkLabel(
            self.prediction_frame,
            text="Prediction will appear here",
            font=ctk.CTkFont(size=18)
        )
        self.prediction_label.pack(pady=10)
        
    def load_model_file(self):
        model_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth")]
        )
        if model_path:
            try:
                # Initialize model and image processor
                self.model = MaizeViTModel(num_classes=6)
                
                # Load the checkpoint
                checkpoint = torch.load(model_path)
                
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # If it's a dictionary with state_dict (from grid search)
                    self.model.vit.load_state_dict(checkpoint['state_dict'])
                    self.model_status.configure(
                        text=f"Grid search model loaded successfully from:\n{model_path}",
                        text_color="green"
                    )
                else:
                    # If it's just the state dict (from regular training)
                    self.model.vit.load_state_dict(checkpoint)
                    self.model_status.configure(
                        text=f"Training model loaded successfully from:\n{model_path}",
                        text_color="green"
                    )
                
                self.model.to(self.device)
                self.model.eval()
                
                self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
                self.image_button.configure(state="normal")
                
            except Exception as e:
                self.model_status.configure(
                    text=f"Error loading model: {str(e)}",
                    text_color="red"
                )
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path and self.model and self.image_processor:
            try:
                # Load and display image
                image = Image.open(file_path)
                # Resize for display
                image.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(image)
                self.display_label.configure(image=photo)
                self.display_label.image = photo
                
                # Make prediction
                self.predict_image(Image.open(file_path).convert('RGB'))
                
            except Exception as e:
                self.prediction_label.configure(
                    text=f"Error processing image: {str(e)}"
                )
    
    def predict_image(self, image):
        try:
            # Prepare image
            inputs = self.image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Update prediction label
            prediction_text = (
                f"Predicted Class: {self.class_names[predicted_class]}\n"
                f"Confidence: {confidence:.2%}"
            )
            self.prediction_label.configure(text=prediction_text)
            
        except Exception as e:
            self.prediction_label.configure(
                text=f"Error during prediction: {str(e)}"
            )
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MaizeClassifierGUI()
    app.run()