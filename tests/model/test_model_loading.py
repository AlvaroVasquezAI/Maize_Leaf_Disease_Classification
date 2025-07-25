import unittest
import torch
import os
import time
import psutil
from transformers import ViTImageProcessor
from src.models.vit_model import MaizeViTModel

class TestModelLoading(unittest.TestCase):
    def setUp(self):
        self.model_path = r"models/grid_search/best_model/best_model_20250227_133601.pth"
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        
    def test_model_load(self):
        """Verificar carga correcta del modelo."""
        # Iniciar temporizador
        start_time = time.time()
        
        # Cargar modelo
        model = MaizeViTModel(num_classes=6)
        
        # Verificar carga de pesos
        checkpoint = torch.load(self.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.vit.load_state_dict(checkpoint['state_dict'])
        else:
            model.vit.load_state_dict(checkpoint)
            
        # Medir tiempo
        load_time = time.time() - start_time
        
        # Verificar que el modelo se cargó correctamente
        self.assertIsNotNone(model.vit)
        self.assertTrue(hasattr(model.vit, 'classifier'))
        self.assertTrue(hasattr(model.vit, 'vit'))
        
        # Medir uso de memoria
        memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2 - self.start_memory
        
        print(f"\nResultados de carga del modelo:")
        print(f"✓ Modelo cargado correctamente")
        print(f"✓ Tiempo de carga: {load_time:.2f} segundos")
        print(f"✓ Memoria utilizada: {memory_used:.2f} MB")
        
    def test_model_eval_mode(self):
        """Verificar que el modelo está en modo eval."""
        model = MaizeViTModel(num_classes=6)
        checkpoint = torch.load(self.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.vit.load_state_dict(checkpoint['state_dict'])
        else:
            model.vit.load_state_dict(checkpoint)
        
        model.eval()
        
        # Verificar que el modelo está en modo eval
        self.assertFalse(model.vit.training)
        print(f"✓ Modelo configurado correctamente en modo eval")
    
    def test_device_selection(self):
        """Verificar selección correcta de dispositivo."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Verificar que el dispositivo se selecciona correctamente
        if torch.cuda.is_available():
            self.assertEqual(device.type, 'cuda')
            print(f"✓ Dispositivo GPU detectado: {torch.cuda.get_device_name(0)}")
            print(f"✓ Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.assertEqual(device.type, 'cpu')
            print(f"✓ Dispositivo CPU detectado")

if __name__ == '__main__':
    unittest.main()