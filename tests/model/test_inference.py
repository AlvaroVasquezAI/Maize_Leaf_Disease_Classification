import unittest
import torch
import time
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor
from src.models.vit_model import MaizeViTModel
import matplotlib.pyplot as plt
import gc

class TestInference(unittest.TestCase):
    def setUp(self):
        self.model_path = r"models/grid_search/best_model/best_model_20250227_133601.pth"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar modelo
        self.model = MaizeViTModel(num_classes=6)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.vit.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.vit.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Cargar procesador de imágenes
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        # Cargar imagen de prueba real (idealmente de cada clase)
        self.test_images = []
        for disease in ["Common_Rust", "Gray_Leaf_Spot", "Healthy", 
                        "Northern_Leaf_Blight", "Phaeosphaeria_Leaf_Spot", "Southern_Rust"]:
            try:
                # Aquí deberías poner la ruta a una imagen de prueba para cada enfermedad
                img_path = f"tests/test_images/{disease}/{disease}_test_01.jpg"
                self.test_images.append((disease, Image.open(img_path).convert('RGB')))
            except Exception as e:
                # Si no encuentra la imagen, crear una imagen de prueba sintética
                print(f"No se pudo cargar imagen para {disease}, usando imagen sintética")
                img = Image.new('RGB', (224, 224), color='green')
                self.test_images.append((disease, img))
        
        # Nombres de clases
        self.class_names = [
            'Common Rust',
            'Gray Leaf Spot',
            'Healthy',
            'Northern Leaf Blight',
            'Phaeosphaeria Leaf Spot',
            'Southern Rust'
        ]
    
    def test_inference_time(self):
        """Verificar tiempo de inferencia."""
        if not self.test_images:
            self.skipTest("No hay imágenes de prueba disponibles")
        
        # Usar la primera imagen para la prueba
        test_image = self.test_images[0][1]
        
        # Preparar la imagen
        inputs = self.image_processor(images=test_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Medir tiempo sin contar carga de modelo
        times = []
        for _ in range(10):  # Hacer múltiples ejecuciones para obtener un promedio
            # Limpiar caché
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Añadir sincronización para medición precisa en GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        print(f"\nResultados de tiempo de inferencia:")
        print(f"✓ Tiempo promedio de inferencia: {avg_time*1000:.2f} ms")
        print(f"✓ Tiempo mínimo: {min(times)*1000:.2f} ms")
        print(f"✓ Tiempo máximo: {max(times)*1000:.2f} ms")
        print(f"✓ Dispositivo: {self.device}")
    
    def test_output_format(self):
        """Verificar formato de salida."""
        if not self.test_images:
            self.skipTest("No hay imágenes de prueba disponibles")
        
        # Usar la primera imagen para la prueba
        test_image = self.test_images[0][1]
        
        # Preparar la imagen
        inputs = self.image_processor(images=test_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verificar que outputs tiene logits
        self.assertTrue(hasattr(outputs, 'logits'))
        
        # Verificar forma de logits (batch_size, num_classes)
        self.assertEqual(outputs.logits.shape, (1, len(self.class_names)))
        
        print(f"✓ Formato de salida correcto")
        print(f"✓ Forma de los logits: {outputs.logits.shape}")
    
    def test_probability_range(self):
        """Verificar rango de probabilidades (0-1)."""
        if not self.test_images:
            self.skipTest("No hay imágenes de prueba disponibles")
        
        # Usar la primera imagen para la prueba
        test_image = self.test_images[0][1]
        
        # Preparar la imagen
        inputs = self.image_processor(images=test_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Verificar que las probabilidades suman aproximadamente 1
        self.assertAlmostEqual(probabilities.sum().item(), 1.0, places=5)
        
        # Verificar que todas las probabilidades están entre 0 y 1
        self.assertTrue((probabilities >= 0).all().item())
        self.assertTrue((probabilities <= 1).all().item())
        
        print(f"✓ Rango de probabilidades correcto (0-1)")
        print(f"✓ Suma de probabilidades: {probabilities.sum().item():.5f}")
    
    # Continuación de test_inference.py
    def test_prediction_consistency(self):
        """Verificar consistencia de predicciones."""
        if not self.test_images:
            self.skipTest("No hay imágenes de prueba disponibles")
        
        # Usar la primera imagen para la prueba
        test_image = self.test_images[0][1]
        
        # Preparar la imagen
        inputs = self.image_processor(images=test_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Hacer múltiples predicciones
        predictions = []
        for _ in range(5):
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                predictions.append(predicted_class)
            
            # Limpiar caché entre ejecuciones
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        # Verificar que todas las predicciones son iguales
        self.assertEqual(len(set(predictions)), 1, "Las predicciones no son consistentes")
        
        print(f"✓ Predicciones consistentes en múltiples ejecuciones")
        print(f"✓ Clase predicha: {self.class_names[predictions[0]]}")
    
    def test_memory_usage(self):
        """Verificar uso de memoria durante inferencia."""
        if not torch.cuda.is_available():
            self.skipTest("Esta prueba requiere GPU")
        
        if not self.test_images:
            self.skipTest("No hay imágenes de prueba disponibles")
        
        # Limpiar memoria antes de empezar
        torch.cuda.empty_cache()
        gc.collect()
        
        # Medir memoria inicial
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Preparar la imagen
        test_image = self.test_images[0][1]
        inputs = self.image_processor(images=test_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Realizar inferencia
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Medir memoria después de inferencia
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        # Calcular uso de memoria
        memory_used = peak_memory - initial_memory
        
        print(f"\nResultados de uso de memoria:")
        print(f"✓ Memoria pico durante inferencia: {peak_memory/1024**2:.2f} MB")
        print(f"✓ Memoria adicional utilizada: {memory_used/1024**2:.2f} MB")
        print(f"✓ Memoria actual después de inferencia: {current_memory/1024**2:.2f} MB")
        
        # Limpiar memoria al finalizar
        torch.cuda.empty_cache()
    
    def test_multi_class_prediction(self):
        """Verificar predicciones en múltiples clases."""
        results = []
        
        for class_name, test_image in self.test_images:
            # Preparar la imagen
            inputs = self.image_processor(images=test_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Realizar inferencia
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Guardar resultados
            results.append({
                'real_class': class_name,
                'predicted_class': self.class_names[predicted_class],
                'confidence': confidence
            })
            
        # Imprimir resultados
        print("\nResultados de predicción multi-clase:")
        print(f"{'Clase Real':<25} {'Clase Predicha':<25} {'Confianza':<10}")
        print("-" * 60)
        for result in results:
            print(f"{result['real_class']:<25} {result['predicted_class']:<25} {result['confidence']:.4f}")

if __name__ == '__main__':
    unittest.main()