import unittest
import requests
import time
import os
import json
import threading
import socketserver
import http.server
import random
import numpy as np
from PIL import Image, ImageDraw
import io

class TestNetwork(unittest.TestCase):
    def setUp(self):
        # URL base de la API
        self.base_url = "http://localhost:5000"
        
        # Crear directorios para resultados
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("test_results/network", exist_ok=True)
        
        # Preparar imágenes de prueba
        self.prepare_test_images()
    
    def prepare_test_images(self):
        """Preparar imágenes para pruebas de red."""
        os.makedirs('test_images', exist_ok=True)
        
        self.test_images = []
        
        # Crear una imagen para pruebas si no existe
        img_path = 'test_images/network_test.jpg'
        
        if not os.path.exists(img_path):
            # Crear imagen con patrón para simular hoja enferma
            img = Image.new('RGB', (224, 224), color=(100, 150, 100))
            draw = ImageDraw.Draw(img)
            
            # Dibujar manchas que simulan enfermedad
            for i in range(20):
                x = random.randint(0, 200)
                y = random.randint(0, 200)
                radius = random.randint(5, 15)
                draw.ellipse((x, y, x+radius, y+radius), fill=(139, 69, 19))
            
            img.save(img_path)
        
        self.test_images.append(img_path)
    
    def test_reconnection(self):
        """Prueba de reconexión después de perder la conexión."""
        print("\nPrueba de reconexión después de perder la conexión")
        
        test_image = self.test_images[0]
        
        # Paso 1: Verificar que el servidor está activo
        try:
            response = requests.get(f"{self.base_url}/health")
            self.assertEqual(response.status_code, 200)
            print("✓ Servidor activo inicialmente")
        except Exception as e:
            self.fail(f"El servidor no está activo: {str(e)}")
        
        # Paso 2: Enviar solicitud inicial
        try:
            img_file = open(test_image, 'rb')
            try:
                files = {'image': (os.path.basename(test_image), img_file, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/predict", files=files)
            finally:
                img_file.close()
            
            self.assertEqual(response.status_code, 200)
            initial_result = response.json()
            print(f"✓ Solicitud inicial exitosa: {initial_result['className']} ({initial_result['confidence']:.4f})")
        except Exception as e:
            self.fail(f"Error en solicitud inicial: {str(e)}")
        
        # Paso 3: Simular pérdida de conexión
        # Nota: En un entorno real, esto podría hacerse apagando temporalmente el servidor
        # Aquí lo simularemos con un tiempo de espera y un endpoint inválido
        print("\nSimulando pérdida de conexión...")
        
        for i in range(3):
            try:
                # Intentar conectarse a un endpoint que no existe
                response = requests.get(f"{self.base_url}/nonexistent", timeout=1)
            except:
                # Se espera que falle
                pass
            time.sleep(1)
            print(f"Intento fallido {i+1}/3")
        
        # Paso 4: Intentar reconexión
        print("\nIntentando reconexión...")
        reconnect_time = None
        reconnect_success = False
        
        for i in range(5):  # Intentar hasta 5 veces
            try:
                start_time = time.time()

                img_file = open(test_image, 'rb')
                try:
                    files = {'image': (os.path.basename(test_image), img_file, 'image/jpeg')}
                    response = requests.post(f"{self.base_url}/predict", files=files, timeout=5)
                finally:
                    img_file.close()
                
                if response.status_code == 200:
                    reconnect_time = time.time() - start_time
                    reconnect_result = response.json()
                    reconnect_success = True
                    print(f"✓ Reconexión exitosa en intento {i+1}")
                    print(f"✓ Tiempo de reconexión: {reconnect_time:.2f} segundos")
                    print(f"✓ Resultado: {reconnect_result['className']} ({reconnect_result['confidence']:.4f})")
                    break
            except Exception as e:
                print(f"Intento de reconexión {i+1} fallido: {str(e)}")
            
            time.sleep(2)  # Esperar antes del siguiente intento
        
        # Paso 5: Verificar consistencia después de reconexión
        if reconnect_success:
            # Las clases deben ser consistentes antes y después de la reconexión
            self.assertEqual(initial_result['className'], reconnect_result['className'])
            
            # La confianza debe ser similar (no exactamente igual debido a variaciones en procesamiento)
            confidence_diff = abs(initial_result['confidence'] - reconnect_result['confidence'])
            self.assertLess(confidence_diff, 0.1)
            
            print(f"✓ Consistencia después de reconexión: Misma clase, diferencia de confianza {confidence_diff:.4f}")
        else:
            self.fail("No se pudo reconectar después de simular pérdida de conexión")
        
        # Guardamos los resultados
        # Guardamos los resultados (modo escritura, crea o trunca el fichero)
        with open('test_results/network/reconnection_test.json', 'w') as f:
            json.dump({
                'initial_result': initial_result,
                'reconnect_success': reconnect_success,
                'reconnect_time': reconnect_time,
                'reconnect_result': reconnect_result if reconnect_success else None,
                'confidence_diff': confidence_diff if reconnect_success else None
            }, f, indent=2)
    
    def test_timeout_handling(self):
        """Prueba de manejo de timeouts en la conexión."""
        print("\nPrueba de manejo de timeouts en la conexión")
        
        test_image = self.test_images[0]
        
        # Probar con diferentes timeouts
        timeout_values = [0.1, 0.5, 1.0, 5.0, 10.0]
        results = []
        
        for timeout in timeout_values:
            print(f"\nProbando con timeout = {timeout} segundos")
            
            try:
                img_file = open(test_image, 'rb')
                try:
                    files = {'image': (os.path.basename(test_image), img_file, 'image/jpeg')}
                    
                    start_time = time.time()
                    response = requests.post(
                        f"{self.base_url}/predict",
                        files=files,
                        timeout=timeout
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        result = {
                            'timeout': timeout,
                            'success': True,
                            'response_time': end_time - start_time,
                            'server_time': response_data['processingTime'],
                            'class': response_data['className'],
                            'confidence': response_data['confidence']
                        }
                        
                        print(f"✓ Éxito: Respuesta en {(end_time-start_time)*1000:.2f} ms")
                        print(f"✓ Clase: {response_data['className']} ({response_data['confidence']:.4f})")
                    else:
                        result = {
                            'timeout': timeout,
                            'success': False,
                            'response_time': end_time - start_time,
                            'status_code': response.status_code,
                            'error': response.text
                        }
                        
                        print(f"✗ Error: {response.status_code} en {(end_time-start_time)*1000:.2f} ms")
                finally:
                    img_file.close()

            except requests.exceptions.Timeout:
                end_time = time.time()
                result = {
                    'timeout': timeout,
                    'success': False,
                    'response_time': end_time - start_time,
                    'error': 'Timeout'
                }
                
                print(f"✗ Error: Timeout después de {(end_time-start_time)*1000:.2f} ms")
            
            except Exception as e:
                result = {
                    'timeout': timeout,
                    'success': False,
                    'error': str(e)
                }
                
                print(f"✗ Error: {str(e)}")
            
            results.append(result)
        
        # Análisis de resultados
        success_count = sum(1 for r in results if r.get('success', False))
        
        print("\n" + "="*50)
        print("RESULTADOS DE PRUEBA DE TIMEOUT")
        print("="*50)
        
        print(f"\nTotal de pruebas: {len(results)}")
        print(f"Pruebas exitosas: {success_count}")
        
        for result in results:
            timeout = result['timeout']
            status = "✓ Éxito" if result.get('success', False) else "✗ Error"
            
            if result.get('success', False):
                details = f"Tiempo: {result['response_time']*1000:.2f} ms"
            else:
                details = f"Error: {result.get('error', 'Desconocido')}"
            
            print(f"Timeout {timeout}s: {status} - {details}")
        
        # Tiempo crítico: el timeout mínimo necesario para una respuesta exitosa
        successful_timeouts = [r['timeout'] for r in results if r.get('success', False)]
        if successful_timeouts:
            critical_timeout = min(successful_timeouts)
            print(f"\nTimeout crítico: {critical_timeout} segundos")
        
        # Guardar resultados
        # Guardar resultados (modo escritura)
        with open('test_results/network/timeout_test.json', 'w') as f:
            json.dump({
                'timeouts_tested': timeout_values,
                'success_count': success_count,
                'critical_timeout': critical_timeout if successful_timeouts else None,
                'detailed_results': results
            }, f, indent=2)
        
        # Verificación final
        self.assertGreater(success_count, 0, "Debería haber al menos una solicitud exitosa")
    
    def test_error_messages(self):
        """Prueba de mensajes de error en diferentes escenarios de red."""
        print("\nPrueba de mensajes de error en diferentes escenarios de red")
        
        scenarios = [
            {
                'name': 'Archivo no válido',
                'setup': lambda: (lambda f: (f.write('This is not an image'), f.close()))(open('test_results/network/invalid.txt', 'w')),
                'files': lambda: {'image': ('invalid.txt', open('test_results/network/invalid.txt', 'rb'), 'text/plain')},
                'expected_status': 500
            },
            {
                'name': 'Sin archivo',
                'setup': lambda: None,
                'files': lambda: {},
                'expected_status': 400
            },
            {
                'name': 'Timeout corto',
                'setup': lambda: None,
                'files': lambda: {'image': (os.path.basename(self.test_images[0]), open(self.test_images[0], 'rb'), 'image/jpeg')},
                'timeout': 0.001,
                'expected_error': 'timeout'
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\nEscenario: {scenario['name']}")
            
            try:
                # Configuración del escenario
                if scenario.get('setup'):
                    scenario['setup']()
                
                # Preparar archivos
                files = scenario['files']()
                
                # Enviar solicitud
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.base_url}/predict",
                        files=files,
                        timeout=scenario.get('timeout', 30)
                    )
                    end_time = time.time()
                    
                    result = {
                        'scenario': scenario['name'],
                        'status_code': response.status_code,
                        'response_time': end_time - start_time,
                        'expected_status': scenario.get('expected_status'),
                        'response_text': response.text[:200]  # Limitar longitud
                    }
                    
                    if 'expected_status' in scenario:
                        if response.status_code == scenario['expected_status']:
                            print(f"✓ Status code correcto: {response.status_code}")
                        else:
                            print(f"✗ Status code incorrecto: {response.status_code}, esperado: {scenario['expected_status']}")
                    
                    print(f"✓ Respuesta: {response.text[:100]}...")
                
                except requests.exceptions.Timeout:
                    result = {
                        'scenario': scenario['name'],
                        'error': 'Timeout',
                        'expected_error': scenario.get('expected_error')
                    }
                    
                    if scenario.get('expected_error') == 'timeout':
                        print(f"✓ Error esperado: Timeout")
                    else:
                        print(f"✗ Error inesperado: Timeout")
                
                except Exception as e:
                    result = {
                        'scenario': scenario['name'],
                        'error': str(e),
                        'expected_error': scenario.get('expected_error')
                    }
                    
                    print(f"✗ Error: {str(e)}")
            
            finally:
                # Limpiar recursos
                if 'invalid.txt' in str(scenario.get('files', '')):
                    try:
                        os.remove('test_results/network/invalid.txt')
                    except:
                        pass
                
                # Cerrar archivos abiertos
                for value in files.values():
                    if isinstance(value, tuple) and len(value) > 1 and hasattr(value[1], 'close'):
                        value[1].close()
            
            results.append(result)
        
        # Análisis de resultados
        print("\n" + "="*50)
        print("RESULTADOS DE PRUEBA DE MENSAJES DE ERROR")
        print("="*50)
        
        for result in results:
            scenario = result['scenario']
            
            if 'status_code' in result and 'expected_status' in result:
                status = "✓ Éxito" if result['status_code'] == result['expected_status'] else "✗ Error"
                details = f"Status: {result['status_code']}, Esperado: {result['expected_status']}"
            elif 'error' in result and 'expected_error' in result:
                status = "✓ Éxito" if result['expected_error'] in result['error'].lower() else "✗ Error"
                details = f"Error: {result['error']}"
            else:
                status = "✗ No verificable"
                details = "Resultado incompleto"
            
            print(f"{scenario}: {status} - {details}")

        # Guardar resultados de error messages (modo escritura)
        with open('test_results/network/error_messages_test.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == '__main__':
    unittest.main()