import unittest
import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor
import matplotlib.pyplot as plt
import io

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        # Crear una imagen de prueba (RGB, 300x300)
        self.test_image = Image.new('RGB', (300, 300), color='green')
        
        # Dibujar algunos elementos en la imagen para simular una hoja
        self.draw_test_pattern(self.test_image)
    
    def draw_test_pattern(self, image):
        """Dibujar un patrón de prueba para simular una hoja con manchas."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        # Dibujar manchas simulando enfermedad
        draw.ellipse((100, 100, 150, 150), fill='brown')
        draw.ellipse((170, 130, 200, 160), fill='brown')
        draw.ellipse((70, 180, 110, 220), fill='brown')
    
    def test_processor_initialization(self):
        """Verificar inicialización del ViT processor."""
        self.assertIsNotNone(self.image_processor)
        self.assertEqual(self.image_processor.size, {'height': 224, 'width': 224})
        print(f"✓ Procesador ViT inicializado correctamente")
        print(f"✓ Configuración del procesador: Tamaño de imagen = {self.image_processor.size}")
    
    def test_image_transformations(self):
        """Verificar transformaciones correctas de la imagen."""
        # Procesar la imagen
        inputs = self.image_processor(images=self.test_image, return_tensors="pt")
        
        # Verificar que la salida tenga el formato correcto
        self.assertIn('pixel_values', inputs)
        
        # Verificar dimensiones
        pixel_values = inputs['pixel_values']
        self.assertEqual(pixel_values.shape[0], 1)  # Batch size de 1
        self.assertEqual(pixel_values.shape[1], 3)  # 3 canales RGB
        self.assertEqual(pixel_values.shape[2], 224)  # Altura 224
        self.assertEqual(pixel_values.shape[3], 224)  # Anchura 224
        
        print(f"✓ Transformación de imagen correcta")
        print(f"✓ Forma del tensor de salida: {pixel_values.shape}")
    
    def test_normalization(self):
        """Verificar normalización de valores."""
        inputs = self.image_processor(images=self.test_image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        
        # Verificar que los valores están normalizados (típicamente entre -1 y 1)
        min_val = pixel_values.min().item()
        max_val = pixel_values.max().item()
        
        # ViT normalmente normaliza con media 0.5 y std 0.5 por canal
        self.assertLess(min_val, 0)
        self.assertGreater(max_val, 0)
        
        print(f"✓ Valores normalizados correctamente")
        print(f"✓ Rango de valores: [{min_val:.4f}, {max_val:.4f}]")
    
    def test_tensor_format(self):
        """Verificar formato de tensor correcto."""
        inputs = self.image_processor(images=self.test_image, return_tensors="pt")
        
        # Verificar que es un tensor de PyTorch
        self.assertIsInstance(inputs['pixel_values'], torch.Tensor)
        
        # Verificar que el dtype es float
        self.assertEqual(inputs['pixel_values'].dtype, torch.float32)
        
        print(f"✓ Formato de tensor correcto: {inputs['pixel_values'].dtype}")
        
    def test_visualize_transformation(self):
        """Visualizar la transformación para verificación manual."""
        # Este test es opcional, para inspección visual
        inputs = self.image_processor(images=self.test_image, return_tensors="pt")
        
        # Convertir el tensor de vuelta a un formato visualizable
        # Desnormalizar y convertir a numpy
        img = inputs['pixel_values'][0].permute(1, 2, 0).detach().numpy()
        # Desnormalizar (aproximadamente - depende de la normalización exacta utilizada)
        img = img * 0.5 + 0.5
        # Asegurar que está en el rango [0, 1]
        img = np.clip(img, 0, 1)
        
        print(f"✓ Transformación visual: Ver tensor convertido a imagen")
        print(f"✓ Forma de la imagen transformada: {img.shape}")

if __name__ == '__main__':
    unittest.main()