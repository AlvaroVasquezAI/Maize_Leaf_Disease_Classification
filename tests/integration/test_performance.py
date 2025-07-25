import unittest
import requests
import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import threading
import io
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from PIL.Image import core as Image
from PIL import ImageDraw

class TestPerformance(unittest.TestCase):
    def setUp(self):
        # URL base de la API
        self.base_url = "http://localhost:5000"
        
        # Crear directorio para resultados
        os.makedirs("test_results", exist_ok=True)
        
        # Preparar imágenes para pruebas
        self.test_images = self.prepare_test_images()
    
    def prepare_test_images(self):
        """Preparar imágenes para pruebas de carga utilizando imágenes reales."""
        # Buscar imágenes existentes o crear nuevas
        if os.path.exists('test_images') and len(os.listdir('test_images')) > 0:
            # Usar imágenes existentes
            image_files = [os.path.join('test_images', f) for f in os.listdir('test_images') 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                print(f"Utilizando {len(image_files)} imágenes existentes de test_images/")
                return image_files
        
        # Intentar encontrar imágenes reales en las carpetas de datos
        real_images = []
        possible_data_paths = [
            'data/test',  # Directorio principal de test
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/test')  # Ruta relativa
        ]
        
        for base_path in possible_data_paths:
            if os.path.exists(base_path):
                print(f"Buscando imágenes en {base_path}")
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            real_images.append(os.path.join(root, file))
                            if len(real_images) >= 5:  # Limitar a 5 imágenes
                                break
                    if len(real_images) >= 5:
                        break
                if real_images:
                    print(f"Encontradas {len(real_images)} imágenes reales para pruebas")
                    return real_images
        
        # Si no se encuentran imágenes reales, crear sintéticas como respaldo
        print("No se encontraron imágenes reales, creando imágenes sintéticas de respaldo")
        os.makedirs('test_images', exist_ok=True)
        images = []
        
        # Crear 5 imágenes diferentes
        for i in range(5):
            # Crear imagen base
            img = Image.new('RGB', (224, 224), color=(100, 150, 100))
            
            # Añadir algún patrón para que sean diferentes
            draw = ImageDraw.Draw(img)
            for j in range(10):
                x1 = np.random.randint(0, 200)
                y1 = np.random.randint(0, 200)
                x2 = x1 + np.random.randint(10, 50)
                y2 = y1 + np.random.randint(10, 50)
                draw.rectangle([x1, y1, x2, y2], fill=(139, 69, 19))
            
            # Guardar imagen
            img_path = f'test_images/test_perf_{i}.jpg'
            img.save(img_path)
            images.append(img_path)
        
        print(f"Creadas {len(images)} imágenes sintéticas para pruebas")
        return images
    
    def test_server_load(self):
        """Probar el rendimiento del servidor bajo carga."""
        print("\nPrueba de carga del servidor")
        
        # Parámetros de la prueba
        num_requests = 100  # Número total de solicitudes
        max_workers = 5    # Número máximo de solicitudes concurrentes
        
        print(f"Configuración: {num_requests} solicitudes, {max_workers} trabajadores concurrentes")
        
        # Resultados
        response_times = []
        success_count = 0
        error_count = 0
        
        # Función para enviar una solicitud
        def send_request(img_path):
            nonlocal success_count, error_count
            
            try:
                with open(img_path, 'rb') as img_file:
                    files = {'image': (os.path.basename(img_path), img_file, 'image/jpeg')}
                    
                    start_time = time.time()
                    response = requests.post(f"{self.base_url}/predict", files=files)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        success_count += 1
                        return {
                            'success': True,
                            'response_time': end_time - start_time,
                            'server_time': response.json()['processingTime'],
                            'confidence': response.json()['confidence'],
                            # Continuación de test_performance.py
                            'class': response.json()['className']
                        }
                    else:
                        error_count += 1
                        return {
                            'success': False,
                            'status_code': response.status_code,
                            'error': response.text
                        }
            except Exception as e:
                error_count += 1
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # Generar lista de imágenes para enviar (repetidas si es necesario)
        request_images = []
        for i in range(num_requests):
            request_images.append(self.test_images[i % len(self.test_images)])
        
        # Enviar solicitudes concurrentes
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f"Enviando {num_requests} solicitudes...")
            start_time = time.time()
            futures = [executor.submit(send_request, img) for img in request_images]
            
            # Recopilar resultados
            for future in futures:
                result = future.result()
                results.append(result)
                
                # Mostrar progreso
                completed = len(results)
                if completed % 5 == 0 or completed == num_requests:
                    print(f"Completadas {completed}/{num_requests} solicitudes ({completed/num_requests*100:.1f}%)")
        
        # Tiempo total
        total_time = time.time() - start_time
        
        # Calcular estadísticas
        successful_results = [r for r in results if r['success']]
        if successful_results:
            response_times = [r['response_time'] for r in successful_results]
            server_times = [r['server_time'] for r in successful_results]
            
            avg_response_time = sum(response_times) / len(response_times)
            avg_server_time = sum(server_times) / len(server_times)
            
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Calcular percentiles
            p50_response = np.percentile(response_times, 50)
            p90_response = np.percentile(response_times, 90)
            p95_response = np.percentile(response_times, 95)
            
            # Calcular rendimiento en solicitudes por segundo
            rps = num_requests / total_time
            
            # Tasa de error
            error_rate = error_count / num_requests * 100
        
        # Mostrar resultados
        print("\n" + "="*50)
        print("RESULTADOS DE PRUEBA DE CARGA")
        print("="*50)
        
        print(f"\nSolicitudes totales: {num_requests}")
        print(f"Solicitudes exitosas: {success_count}")
        print(f"Solicitudes fallidas: {error_count}")
        print(f"Tasa de error: {error_rate:.2f}%")
        
        if successful_results:
            print(f"\nTiempo total de prueba: {total_time:.2f} segundos")
            print(f"Rendimiento: {rps:.2f} solicitudes por segundo")
            
            print(f"\nTiempos de respuesta:")
            print(f"- Promedio: {avg_response_time*1000:.2f} ms")
            print(f"- Mínimo: {min_response_time*1000:.2f} ms")
            print(f"- Máximo: {max_response_time*1000:.2f} ms")
            print(f"- P50: {p50_response*1000:.2f} ms")
            print(f"- P90: {p90_response*1000:.2f} ms")
            print(f"- P95: {p95_response*1000:.2f} ms")
            
            print(f"\nTiempo promedio de procesamiento en servidor: {avg_server_time*1000:.2f} ms")
        
        # Guardar resultados
        if successful_results:
            with open('test_results/load_test_results.json', 'w') as f:
                json.dump({
                    'config': {
                        'num_requests': num_requests,
                        'max_workers': max_workers
                    },
                    'summary': {
                        'success_count': success_count,
                        'error_count': error_count,
                        'error_rate': error_rate,
                        'total_time': total_time,
                        'rps': rps,
                        'avg_response_time': avg_response_time,
                        'min_response_time': min_response_time,
                        'max_response_time': max_response_time,
                        'p50_response': p50_response,
                        'p90_response': p90_response,
                        'p95_response': p95_response,
                        'avg_server_time': avg_server_time
                    },
                    'detailed_results': results
                }, f, indent=2)
            
            # Generar gráficos
            self.generate_load_test_graphs(response_times, server_times)
        
        # Verificaciones
        self.assertGreater(success_count, 0, "No hubo solicitudes exitosas")
        self.assertLess(error_rate, 50, "Tasa de error mayor al 50%")
        if successful_results:
            self.assertLess(p95_response, 10, "El 95% de las respuestas deben ser más rápidas que 10 segundos")
    
    def test_stability(self):
        """Prueba de estabilidad del servidor durante un periodo prolongado."""
        print("\nPrueba de estabilidad del servidor (duración: 120 segundos)")
        
        # Configuración
        test_duration = 120  # segundos
        interval = 5         # intervalo entre solicitudes (segundos)
        
        # Seleccionar una imagen para pruebas repetidas
        test_image = self.test_images[0]
        
        # Resultados
        results = []
        start_test_time = time.time()
        end_test_time = start_test_time + test_duration
        
        print(f"Iniciando prueba a las {time.strftime('%H:%M:%S')}")
        print(f"Duración programada: {test_duration} segundos")
        print(f"Intervalo entre solicitudes: {interval} segundos")
        
        # Enviar solicitudes periódicas
        while time.time() < end_test_time:
            try:
                # Enviar solicitud
                with open(test_image, 'rb') as img_file:
                    files = {'image': (os.path.basename(test_image), img_file, 'image/jpeg')}
                    
                    start_time = time.time()
                    response = requests.post(f"{self.base_url}/predict", files=files)
                    response_time = time.time() - start_time
                
                # Registrar resultado
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        'timestamp': time.time(),
                        'success': True,
                        'response_time': response_time,
                        'server_time': data['processingTime'],
                        'class': data['className'],
                        'confidence': data['confidence']
                    })
                    
                    elapsed = time.time() - start_test_time
                    print(f"[{elapsed:.1f}s] Respuesta: {data['className']} ({data['confidence']:.4f}), Tiempo: {response_time*1000:.2f} ms")
                else:
                    results.append({
                        'timestamp': time.time(),
                        'success': False,
                        'response_time': response_time,
                        'status_code': response.status_code,
                        'error': response.text
                    })
                    
                    elapsed = time.time() - start_test_time
                    print(f"[{elapsed:.1f}s] Error: {response.status_code}")
            
            except Exception as e:
                results.append({
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e)
                })
                
                elapsed = time.time() - start_test_time
                print(f"[{elapsed:.1f}s] Excepción: {str(e)}")
            
            # Esperar al siguiente intervalo
            time.sleep(interval)
        
        # Calcular duración real
        actual_duration = time.time() - start_test_time
        
        # Estadísticas
        success_count = sum(1 for r in results if r.get('success', False))
        success_rate = success_count / len(results) * 100 if results else 0
        
        if success_count > 0:
            response_times = [r['response_time'] for r in results if r.get('success', False)]
            avg_response_time = sum(response_times) / len(response_times)
            
            # Verificar consistencia de clase predicha
            classes = [r['class'] for r in results if r.get('success', False)]
            most_common_class = max(set(classes), key=classes.count)
            class_consistency = classes.count(most_common_class) / len(classes) * 100
            
            # Verificar drift en tiempos de respuesta
            first_half = response_times[:len(response_times)//2]
            second_half = response_times[len(response_times)//2:]
            
            avg_first_half = sum(first_half) / len(first_half) if first_half else 0
            avg_second_half = sum(second_half) / len(second_half) if second_half else 0
            
            time_drift = (avg_second_half - avg_first_half) / avg_first_half * 100 if avg_first_half else 0
        
        # Mostrar resultados
        print("\n" + "="*50)
        print("RESULTADOS DE PRUEBA DE ESTABILIDAD")
        print("="*50)
        
        print(f"\nDuración real: {actual_duration:.2f} segundos")
        print(f"Solicitudes totales: {len(results)}")
        print(f"Solicitudes exitosas: {success_count}")
        print(f"Tasa de éxito: {success_rate:.2f}%")
        
        if success_count > 0:
            print(f"\nTiempo de respuesta promedio: {avg_response_time*1000:.2f} ms")
            print(f"Clase más común: {most_common_class}")
            print(f"Consistencia de clase: {class_consistency:.2f}%")
            print(f"\nDrift en tiempo de respuesta: {time_drift:.2f}%")
            
            if abs(time_drift) > 10:
                print("⚠️ ADVERTENCIA: Drift significativo en tiempos de respuesta")
            else:
                print("✓ No se detectó drift significativo")
        
        # Guardar resultados
        with open('test_results/stability_test_results.json', 'w') as f:
            json.dump({
                'config': {
                    'test_duration': test_duration,
                    'actual_duration': actual_duration,
                    'interval': interval
                },
                'summary': {
                    'total_requests': len(results),
                    'success_count': success_count,
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time if success_count > 0 else None,
                    'most_common_class': most_common_class if success_count > 0 else None,
                    'class_consistency': class_consistency if success_count > 0 else None,
                    'time_drift': time_drift if success_count > 0 else None
                },
                'detailed_results': results
            }, f, indent=2)
        
        # Generar gráficos
        if success_count > 0:
            self.generate_stability_graphs(results)
        
        # Verificaciones
        self.assertGreater(success_rate, 90, "Tasa de éxito menor al 90%")
        if success_count > 0:
            self.assertGreater(class_consistency, 80, "Consistencia de clase menor al 80%")
            self.assertLess(abs(time_drift), 20, "Drift en tiempo de respuesta mayor al 20%")
    
    def test_network_connectivity(self):
        """Prueba de conectividad de red en diferentes escenarios."""
        print("\nPrueba de conectividad de red")
        
        scenarios = [
            {
                'name': 'Conexión estable',
                'delay': 0,
                'timeout': 30
            },
            {
                'name': 'Conexión lenta',
                'delay': 2,
                'timeout': 30
            },
            {
                'name': 'Timeout corto',
                'delay': 0,
                'timeout': 1
            }
        ]
        
        # Seleccionar una imagen para pruebas
        test_image = self.test_images[0]
        
        results = []
        
        for scenario in scenarios:
            print(f"\nEscenario: {scenario['name']}")
            print(f"Configuración: Delay={scenario['delay']}s, Timeout={scenario['timeout']}s")
            
            try:
                with open(test_image, 'rb') as img_file:
                    files = {'image': (os.path.basename(test_image), img_file, 'image/jpeg')}
                    
                    # Simular delay si se especifica
                    if scenario['delay'] > 0:
                        time.sleep(scenario['delay'])
                    
                    # Enviar solicitud con timeout específico
                    start_time = time.time()
                    response = requests.post(
                        f"{self.base_url}/predict", 
                        files=files,
                        timeout=scenario['timeout']
                    )
                    end_time = time.time()
                    
                    # Procesar respuesta
                    if response.status_code == 200:
                        data = response.json()
                        result = {
                            'scenario': scenario['name'],
                            'success': True,
                            'response_time': end_time - start_time,
                            'server_time': data['processingTime'],
                            'status_code': response.status_code
                        }
                        
                        print(f"✓ Éxito: Respuesta en {(end_time-start_time)*1000:.2f} ms")
                    else:
                        result = {
                            'scenario': scenario['name'],
                            'success': False,
                            'response_time': end_time - start_time,
                            'status_code': response.status_code,
                            'error': response.text
                        }
                        
                        print(f"✗ Error: {response.status_code} en {(end_time-start_time)*1000:.2f} ms")
            
            except requests.exceptions.Timeout:
                result = {
                    'scenario': scenario['name'],
                    'success': False,
                    'error': 'Timeout'
                }
                
                print(f"✗ Error: Timeout después de {scenario['timeout']} segundos")
            
            except requests.exceptions.ConnectionError:
                result = {
                    'scenario': scenario['name'],
                    'success': False,
                    'error': 'ConnectionError'
                }
                
                print(f"✗ Error: Conexión rechazada o interrumpida")
            
            except Exception as e:
                result = {
                    'scenario': scenario['name'],
                    'success': False,
                    'error': str(e)
                }
                
                print(f"✗ Error: {str(e)}")
            
            results.append(result)
        
        # Analizar resultados
        success_count = sum(1 for r in results if r.get('success', False))
        
        print("\n" + "="*50)
        print("RESULTADOS DE PRUEBA DE CONECTIVIDAD")
        print("="*50)
        
        for result in results:
            scenario = result['scenario']
            status = "✓ Éxito" if result.get('success', False) else "✗ Error"
            
            if result.get('success', False):
                details = f"Tiempo: {result['response_time']*1000:.2f} ms"
            else:
                details = f"Error: {result.get('error', 'Desconocido')}"
            
            print(f"{scenario}: {status} - {details}")
        
        # Verificaciones
        # En este caso, esperamos que la conexión estable tenga éxito,
        # pero los otros escenarios pueden fallar dependiendo de la configuración
        self.assertTrue(results[0].get('success', False), "La conexión estable debe funcionar")
    
    def generate_load_test_graphs(self, response_times, server_times):
        """Generar gráficos para la prueba de carga."""
        # 1. Histograma de tiempos de respuesta
        plt.figure(figsize=(10, 6))
        plt.hist(np.array(response_times) * 1000, bins=20, alpha=0.7, color='blue')
        plt.axvline(np.mean(response_times) * 1000, color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribución de tiempos de respuesta')
        plt.xlabel('Tiempo de respuesta (ms)')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig('test_results/load_response_histogram.png')
        
        # 2. Histograma de tiempos de procesamiento en servidor
        plt.figure(figsize=(10, 6))
        plt.hist(np.array(server_times) * 1000, bins=20, alpha=0.7, color='green')
        plt.axvline(np.mean(server_times) * 1000, color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribución de tiempos de procesamiento en servidor')
        plt.xlabel('Tiempo de procesamiento (ms)')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig('test_results/load_server_histogram.png')
        
        print("\nGráficos de prueba de carga guardados en la carpeta 'test_results'")
    
    def generate_stability_graphs(self, results):
        """Generar gráficos para la prueba de estabilidad."""
        # Extraer datos
        timestamps = [(r['timestamp'] - results[0]['timestamp']) for r in results if r.get('success', False)]
        response_times = [r['response_time'] * 1000 for r in results if r.get('success', False)]  # ms
        server_times = [r['server_time'] * 1000 for r in results if r.get('success', False)]  # ms
        confidences = [r['confidence'] for r in results if r.get('success', False)]
        
        # 1. Tiempos de respuesta a lo largo del tiempo
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, response_times, 'b-', label='Tiempo de respuesta')
        plt.plot(timestamps, server_times, 'g-', label='Tiempo de servidor')
        
        # Líneas de tendencia
        if len(timestamps) > 1:
            z1 = np.polyfit(timestamps, response_times, 1)
            p1 = np.poly1d(z1)
            plt.plot(timestamps, p1(timestamps), "r--", label='Tendencia de respuesta')
            
            z2 = np.polyfit(timestamps, server_times, 1)
            p2 = np.poly1d(z2)
            plt.plot(timestamps, p2(timestamps), "m--", label='Tendencia de servidor')
        
        plt.title('Tiempos de respuesta durante la prueba de estabilidad')
        plt.xlabel('Tiempo transcurrido (s)')
        plt.ylabel('Tiempo (ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('test_results/stability_response_times.png')
        
        # 2. Nivel de confianza a lo largo del tiempo
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, confidences, 'b-')
        
        # Línea de tendencia
        if len(timestamps) > 1:
            z = np.polyfit(timestamps, confidences, 1)
            p = np.poly1d(z)
            # Continuación de test_performance.py
            plt.plot(timestamps, p(timestamps), "r--", label='Tendencia de confianza')
        
        plt.title('Nivel de confianza durante la prueba de estabilidad')
        plt.xlabel('Tiempo transcurrido (s)')
        plt.ylabel('Confianza')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('test_results/stability_confidence.png')
        
        print("\nGráficos de prueba de estabilidad guardados en la carpeta 'test_results'")

if __name__ == '__main__':
    unittest.main()