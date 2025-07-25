import unittest
import requests
import time
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import io

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        # URL base de la API
        self.base_url = "http://localhost:5000"
        
        # Intentar obtener rutas de imágenes de prueba
        self.test_images = []
        for disease in ["Common_Rust", "Gray_Leaf_Spot", "Healthy", 
                        "Northern_Leaf_Blight", "Phaeosphaeria_Leaf_Spot", "Southern_Rust"]:
            try:
                # Ejemplo de ruta a una imagen de prueba
                img_path = f"data/test/{disease}/{disease}_test_01.jpg"
                if os.path.exists(img_path):
                    self.test_images.append((disease, img_path))
            except Exception as e:
                print(f"No se pudo encontrar imagen para {disease}: {e}")
        
        # Si no hay imágenes reales, crear una imagen de prueba
        if not self.test_images:
            # Crear una imagen temporal para pruebas
            test_img = Image.new('RGB', (224, 224), color='green')
            test_img_path = 'temp_test_image.jpg'
            test_img.save(test_img_path)
            self.test_images.append(('Test', test_img_path))
    
    def test_health_endpoint(self):
        """Verificar que el endpoint de health está funcionando."""
        try:
            response = requests.get(f"{self.base_url}/health")
            
            # Verificar código de estado
            self.assertEqual(response.status_code, 200)
            
            # Verificar estructura JSON
            data = response.json()
            self.assertIn('status', data)
            self.assertIn('modelLoaded', data)
            self.assertIn('processorLoaded', data)
            
            # Verificar que el modelo está cargado
            self.assertEqual(data['status'], 'UP')
            self.assertTrue(data['modelLoaded'])
            self.assertTrue(data['processorLoaded'])
            
            print(f"\nResultados de verificación de health endpoint:")
            print(f"✓ Código de estado: {response.status_code}")
            print(f"✓ Status: {data['status']}")
            print(f"✓ Modelo cargado: {data['modelLoaded']}")
            print(f"✓ Procesador cargado: {data['processorLoaded']}")
            print(f"✓ Dispositivo: {data['device']}")
            
        except requests.exceptions.ConnectionError:
            self.fail("No se pudo conectar al servidor. Asegúrate de que está en ejecución.")
    
    def test_predict_endpoint(self):
        """Verificar el endpoint de predicción."""
        if not self.test_images:
            self.skipTest("No hay imágenes de prueba disponibles")
        
        # Usar la primera imagen para la prueba
        test_class, test_image_path = self.test_images[0]
        
        # Preparar datos para la solicitud
        with open(test_image_path, 'rb') as img_file:
            files = {'image': (os.path.basename(test_image_path), img_file, 'image/jpeg')}
            
            # Medir tiempo de respuesta
            start_time = time.time()
            response = requests.post(f"{self.base_url}/predict", files=files)
            end_time = time.time()
        
        # Verificar código de estado
        self.assertEqual(response.status_code, 200)
        
        # Verificar estructura JSON
        data = response.json()
        self.assertIn('className', data)
        self.assertIn('confidence', data)
        self.assertIn('processingTime', data)
        
        # Verificar tipos de datos
        self.assertIsInstance(data['className'], str)
        self.assertIsInstance(data['confidence'], float)
        self.assertIsInstance(data['processingTime'], float)
        
        # Verificar rangos de valores
        self.assertGreaterEqual(data['confidence'], 0.0)
        self.assertLessEqual(data['confidence'], 1.0)
        self.assertGreater(data['processingTime'], 0.0)
        
        # Calcular tiempo de respuesta total (incluido HTTP)
        response_time = end_time - start_time
        
        print(f"\nResultados de verificación de predict endpoint:")
        print(f"✓ Código de estado: {response.status_code}")
        print(f"✓ Clase predicha: {data['className']}")
        print(f"✓ Confianza: {data['confidence']:.4f}")
        print(f"✓ Tiempo de procesamiento (servidor): {data['processingTime']*1000:.2f} ms")
        print(f"✓ Tiempo de respuesta total (HTTP): {response_time*1000:.2f} ms")
    
    def test_http_error_codes(self):
        """Verificar que la API maneja correctamente los errores."""
        # Prueba 1: Sin imagen
        response = requests.post(f"{self.base_url}/predict")
        self.assertEqual(response.status_code, 400)
        
        # Prueba 2: Imagen no válida
        with open('test_invalid.txt', 'w') as f:
            f.write("This is not an image")
        
        with open('test_invalid.txt', 'rb') as f:
            files = {'image': ('test_invalid.txt', f, 'text/plain')}
            response = requests.post(f"{self.base_url}/predict", files=files)
        
        # Esperar error 500 ya que el servidor intentará procesar un archivo no válido
        self.assertEqual(response.status_code, 500)
        
        # Limpiar
        os.remove('test_invalid.txt')
        
        print(f"\nResultados de verificación de manejo de errores:")
        print(f"✓ Solicitud sin imagen: código {response.status_code}")
        print(f"✓ Solicitud con archivo no válido: código {response.status_code}")
    
    def test_response_times(self):
        """Verificar tiempos de respuesta bajo múltiples solicitudes."""
        if not self.test_images:
            self.skipTest("No hay imágenes de prueba disponibles")
        
        # Usar la primera imagen para pruebas de rendimiento
        test_class, test_image_path = self.test_images[0]
        
        times = []
        for i in range(5):  # Realizar 5 solicitudes
            with open(test_image_path, 'rb') as img_file:
                files = {'image': (os.path.basename(test_image_path), img_file, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post(f"{self.base_url}/predict", files=files)
                end_time = time.time()
                
                self.assertEqual(response.status_code, 200)
                times.append(end_time - start_time)
        
        # Calcular estadísticas
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nResultados de tiempos de respuesta (5 solicitudes):")
        print(f"✓ Tiempo promedio: {avg_time*1000:.2f} ms")
        print(f"✓ Tiempo mínimo: {min_time*1000:.2f} ms")
        print(f"✓ Tiempo máximo: {max_time*1000:.2f} ms")

if __name__ == '__main__':
    unittest.main()