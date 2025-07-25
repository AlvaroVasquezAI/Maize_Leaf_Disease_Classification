import unittest
import requests
import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import io

class TestFullFlow(unittest.TestCase):
    def setUp(self):
        # URL base de la API
        self.base_url = "http://localhost:5000"
        
        # Crear directorio para resultados de pruebas
        os.makedirs("test_results", exist_ok=True)
        
        # Crear imágenes de prueba con diferentes características
        self.setup_test_images()
        
        # Clases de enfermedades
        self.disease_classes = [
            'Common Rust',
            'Gray Leaf Spot',
            'Healthy',
            'Northern Leaf Blight',
            'Phaeosphaeria Leaf Spot',
            'Southern Rust'
        ]
    
    def create_test_images(self):
        """Crear imágenes de prueba simulando hojas con diferentes características."""
        self.test_images = []
        
        # Crear imágenes con diferentes características para pruebas más completas
        os.makedirs('test_images', exist_ok=True)
        
        # Diferentes tamaños y proporciones
        sizes = [(224, 224), (800, 600), (600, 800), (1080, 1080)]
        
        # Colores base para simular diferentes condiciones
        base_colors = [
            (100, 150, 50),  # Verde estándar
            (130, 180, 80),  # Verde claro
            (70, 120, 40),   # Verde oscuro
            (150, 170, 100)  # Verde amarillento
        ]
        
        # Crear imágenes para cada combinación de tamaño y color
        for i, size in enumerate(sizes):
            for j, color in enumerate(base_colors):
                img = Image.new('RGB', size, color=color)
                
                # Añadir manchas para simular enfermedades con diferentes patrones
                self.draw_disease_pattern(img, pattern_type=(i + j) % 3)
                
                # Guardar imagen
                img_path = f'test_images/test_leaf_{i}_{j}.jpg'
                img.save(img_path)
                self.test_images.append(img_path)

    # Reemplaza el método create_test_images con este
    def setup_test_images(self):
        """Configurar imágenes reales para las pruebas."""
        self.test_images = []
        
        # Buscar imágenes reales en las carpetas de datos de test
        base_dirs = [
            'data/test/Common_Rust',
            'data/test/Gray_Leaf_Spot',
            'data/test/Healthy',
            'data/test/Northern_Leaf_Blight',
            'data/test/Phaeosphaeria_Leaf_Spot',
            'data/test/Southern_Rust'
        ]
        
        # Intentar obtener al menos 3 imágenes de cada clase (ajustar según necesidad)
        samples_per_class = 20
        
        print("Buscando imágenes reales para pruebas...")
        
        for base_dir in base_dirs:
            if not os.path.exists(base_dir):
                print(f"Advertencia: Directorio {base_dir} no encontrado")
                continue
                
            class_name = os.path.basename(base_dir)
            print(f"Procesando clase: {class_name}")
            
            # Listar archivos de imagen en el directorio
            image_files = [f for f in os.listdir(base_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Seleccionar un subconjunto (para no procesar miles de imágenes)
            selected_files = image_files[:samples_per_class]
            
            for img_file in selected_files:
                img_path = os.path.join(base_dir, img_file)
                self.test_images.append(img_path)
                print(f"  - Añadida: {img_path}")
        
        if not self.test_images:
            print("No se encontraron imágenes reales, creando imágenes sintéticas como respaldo...")
            self.create_test_images()
        else:
            print(f"Se encontraron {len(self.test_images)} imágenes reales para pruebas")
    
    def draw_disease_pattern(self, image, pattern_type=0):
        """Dibujar patrones que simulan diferentes enfermedades."""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        if pattern_type == 0:
            # Patrón tipo Roya (manchas circulares marrones)
            for _ in range(20):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                radius = np.random.randint(5, 15)
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(139, 69, 19))
        
        elif pattern_type == 1:
            # Patrón tipo Mancha gris (manchas rectangulares)
            for _ in range(15):
                x = np.random.randint(0, width-30)
                y = np.random.randint(0, height-30)
                w = np.random.randint(20, 40)
                h = np.random.randint(20, 40)
                draw.rectangle((x, y, x+w, y+h), fill=(120, 120, 120))
        
        else:
            # Patrón tipo Tizón (manchas irregulares)
            for _ in range(10):
                x = np.random.randint(0, width-50)
                y = np.random.randint(0, height-50)
                points = []
                for i in range(6):
                    dx = np.random.randint(0, 50)
                    dy = np.random.randint(0, 50)
                    points.append((x+dx, y+dy))
                draw.polygon(points, fill=(101, 67, 33))
    
    def tearDown(self):
        """Limpiar archivos temporales."""
        # No eliminar los archivos para poder examinarlos después
        pass
    
    def test_full_classification_flow(self):
        """Prueba del flujo completo: captura -> procesamiento -> análisis -> resultado."""
        all_results = []
        
        # Configuración
        print("\nPrueba de flujo completo de clasificación")
        print("Configuración: Procesamiento de imágenes de diferentes tamaños y características")
        print(f"Número de imágenes de prueba: {len(self.test_images)}")
        
        # Procesar cada imagen de prueba
        for i, img_path in enumerate(self.test_images):
            print(f"\nProcesando imagen {i+1}/{len(self.test_images)}: {os.path.basename(img_path)}")
            
            # Obtener propiedades de imagen
            with Image.open(img_path) as img:
                size = img.size
                format = img.format
            
            print(f"Propiedades: Tamaño={size}, Formato={format}")
            
            # Fase 1: Simular captura (aquí utilizamos la imagen existente)
            start_capture = time.time()
            img_file = open(img_path, 'rb')
            capture_time = time.time() - start_capture
            
            print(f"1. Captura simulada: {capture_time*1000:.2f} ms")
            
            # Fase 2: Preprocesamiento (simulado como compresión JPEG)
            start_preprocessing = time.time()
            with Image.open(img_path) as img:
                # Redimensionar si es necesario
                if max(img.size) > 1080:
                    img.thumbnail((1080, 1080))
                
                # Convertir a JPEG con compresión
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=90)
                output.seek(0)
            
            preprocessing_time = time.time() - start_preprocessing
            
            print(f"2. Preprocesamiento: {preprocessing_time*1000:.2f} ms")
            
            # Fase 3: Envío al servidor
            start_upload = time.time()
            files = {'image': ('image.jpg', output, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/predict", files=files)
            upload_time = time.time() - start_upload
            
            print(f"3. Envío al servidor: {upload_time*1000:.2f} ms")
            
            # Fase 4: Procesamiento en servidor (se obtiene de la respuesta)
            if response.status_code == 200:
                data = response.json()
                server_time = data['processingTime'] * 1000  # convertir a ms
                
                print(f"4. Procesamiento en servidor: {server_time:.2f} ms")
                
                # Fase 5: Resultado
                print(f"5. Resultado: {data['className']} (Confianza: {data['confidence']*100:.2f}%)")
                
                # Tiempo total
                total_time = capture_time + preprocessing_time + upload_time + data['processingTime']
                
                # Registrar resultado
                all_results.append({
                    'image': os.path.basename(img_path),
                    'size': size,
                    'capture_time': capture_time,
                    'preprocessing_time': preprocessing_time,
                    'upload_time': upload_time,
                    'server_time': data['processingTime'],
                    'total_time': total_time,
                    'class': data['className'],
                    'confidence': data['confidence']
                })
            else:
                print(f"Error en la solicitud: {response.status_code}")
                print(response.text)
            
            # Limpiar
            if 'img_file' in locals():
                img_file.close()
            
            # Breve pausa para no sobrecargar el servidor
            time.sleep(0.5)
        
        # Resumen de resultados
        print("\n" + "="*50)
        print("RESUMEN DE RESULTADOS DE FLUJO COMPLETO")
        print("="*50)
        
        # Calcular estadísticas
        if all_results:
            # Tiempo promedio por fase
            avg_capture = sum(r['capture_time'] for r in all_results) / len(all_results)
            avg_preproc = sum(r['preprocessing_time'] for r in all_results) / len(all_results)
            avg_upload = sum(r['upload_time'] for r in all_results) / len(all_results)
            avg_server = sum(r['server_time'] for r in all_results) / len(all_results)
            avg_total = sum(r['total_time'] for r in all_results) / len(all_results)
            
            print(f"\nTiempos promedio (ms):")
            print(f"- Captura: {avg_capture*1000:.2f}")
            print(f"- Preprocesamiento: {avg_preproc*1000:.2f}")
            print(f"- Envío: {avg_upload*1000:.2f}")
            print(f"- Servidor: {avg_server*1000:.2f}")
            print(f"- Total: {avg_total*1000:.2f}")
            
            # Confianza promedio
            avg_confidence = sum(r['confidence'] for r in all_results) / len(all_results)
            print(f"\nConfianza promedio: {avg_confidence*100:.2f}%")
            
            # Distribución de clases
            class_counts = {}
            for r in all_results:
                class_name = r['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("\nDistribución de clases:")
            for cls, count in class_counts.items():
                print(f"- {cls}: {count} ({count/len(all_results)*100:.1f}%)")
            
            # Guardar resultados para análisis posterior
            with open('test_results/full_flow_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Generar gráficos
            self.generate_performance_graphs(all_results)
        
        # Verificaciones finales
        self.assertTrue(len(all_results) > 0, "No se obtuvieron resultados")
        self.assertTrue(avg_total < 10, "El tiempo total promedio es mayor a 10 segundos")
        self.assertTrue(avg_confidence > 0.5, "La confianza promedio es menor al 50%")
    
    def generate_performance_graphs(self, results):
                # --- 1. Gráfico de tiempos por fase ---
        # (Asegúrate que 'results' es la lista de diccionarios con los datos)

        if not results:
            print("[AVISO] No hay resultados para generar el gráfico de tiempos por fase.")
            # Si esta función está dentro de una clase de prueba, podrías usar:
            # self.skipTest("No hay resultados para generar el gráfico de tiempos por fase.")
            return # Salir si no hay resultados

        num_images = len(results)
        # Crear una secuencia de números para las posiciones en el eje X (0, 1, 2, ..., num_images-1)
        x_positions = np.arange(num_images)

        # Preparar datos (usando las claves de tu script original)
        # images = [r['image'] for r in results] # Ya no se usa para las posiciones de las barras
        capture_times = np.array([r['capture_time']*1000 for r in results])
        preproc_times = np.array([r['preprocessing_time']*1000 for r in results])
        
        # 'upload_time' es el tiempo de la solicitud POST completa.
        # 'server_time' es el tiempo reportado por el servidor.
        # Para apilar como en tu imagen (donde "Envío" es la barra verde grande):
        #   Barra "Envío" = upload_time - server_time (esto es la red)
        #   Luego se apila "Servidor" encima.
        upload_times_from_dict = np.array([r['upload_time']*1000 for r in results])
        server_times_from_dict = np.array([r['server_time']*1000 for r in results])
        
        network_component_for_bar = upload_times_from_dict - server_times_from_dict
        network_component_for_bar = np.maximum(0, network_component_for_bar) # Evitar negativos

        # --- MANTENEMOS TU FIGSIZE ORIGINAL ---
        plt.figure(figsize=(10, 6)) # Tu figsize original
        
        # --- MANTENEMOS TU BAR_WIDTH ORIGINAL ---
        bar_width = 0.8 # Tu bar_width original
        
        # Crear gráfico de barras apiladas usando x_positions
        # El orden de apilamiento y los colores deben coincidir con tu gráfico original
        # Asumiendo que los colores por defecto C0, C1, C2, C3 coinciden con tu gráfico:
        # C0: azul (Captura), C1: naranja (Preprocesamiento), C2: verde (Envío/Red), C3: rojo (Servidor)

        # bars = plt.bar(images, capture_times, bar_width, label='Captura') # Línea original
        plt.bar(x_positions, capture_times, bar_width, label='Captura') # MODIFICADO: x_positions
        
        # plt.bar(images, preproc_times, bar_width, bottom=capture_times, label='Preprocesamiento') # Línea original
        plt.bar(x_positions, preproc_times, bar_width, bottom=capture_times, label='Preprocesamiento') # MODIFICADO: x_positions
        
        # Calcular posiciones para las siguientes barras
        bottom_after_preproc = capture_times + preproc_times # Correcto
        
        # plt.bar(images, upload_times, bar_width, bottom=bottom, label='Envío') # Línea original
        # La barra "Envío" de tu gráfico original es la componente de red
        plt.bar(x_positions, network_component_for_bar, bar_width, bottom=bottom_after_preproc, label='Envío') # MODIFICADO: x_positions y network_component_for_bar
        
        bottom_after_network = bottom_after_preproc + network_component_for_bar # Correcto
        
        # plt.bar(images, server_times, bar_width, bottom=bottom, label='Servidor') # Línea original
        plt.bar(x_positions, server_times_from_dict, bar_width, bottom=bottom_after_network, label='Servidor') # MODIFICADO: x_positions y server_times_from_dict
        
        plt.title('Tiempos de procesamiento por fase')
        plt.xlabel('Índice de Imagen') # MODIFICADO: Etiqueta del eje X
        plt.ylabel('Tiempo (ms)')
        
        # --- MODIFICACIÓN PRINCIPAL: Etiquetas del eje X ---
        if num_images > 0:
            # Crear etiquetas numéricas del 1 al num_images
            tick_labels = [str(i + 1) for i in x_positions]
            
            # Decidir cuántas etiquetas mostrar para evitar congestión
            if num_images > 30: # Si hay muchas imágenes, mostrar menos ticks
                step = num_images // 15 # Aproximadamente 15 ticks
                selected_ticks = x_positions[::step]
                selected_labels = [tick_labels[i] for i in selected_ticks]
                plt.xticks(selected_ticks, selected_labels, rotation=90, fontsize=8) # Rotar y ajustar fuente
            else: # Mostrar todas si son pocas
                plt.xticks(x_positions, tick_labels, rotation=90, fontsize=8 if num_images > 15 else 10) # Rotar y ajustar fuente
        else:
            plt.xticks([]) # No mostrar ticks si no hay imágenes

        plt.legend()
        plt.tight_layout() # Mantenemos tu tight_layout
        plt.savefig('test_results/processing_times.png')
        plt.close() # Añadido para liberar memoria
        print("[✅] Gráfico 'processing_times.png' generado con índices numéricos en el eje X.")
        
        # 2. Gráfico de confianza por clase
        plt.figure(figsize=(10, 6))
        
        # Agrupar por clase
        class_confidence = {}
        for r in results:
            cls = r['class']
            if cls not in class_confidence:
                class_confidence[cls] = []
            class_confidence[cls].append(r['confidence'])
        
        # Crear boxplot
        data = [class_confidence[cls] for cls in class_confidence]
        plt.boxplot(data, tick_labels=class_confidence.keys())
        plt.title('Distribución de confianza por clase')
        plt.xlabel('Clase')
        plt.ylabel('Confianza')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('test_results/confidence_distribution.png')
        
        # 3. Gráfico de tiempo total vs tamaño de imagen
        plt.figure(figsize=(10, 6))
        
        # Calcular tamaños en megapixels
        sizes = [r['size'][0] * r['size'][1] / 1000000 for r in results]  # megapixels
        total_times = [r['total_time']*1000 for r in results]  # ms
        
        plt.scatter(sizes, total_times)
        plt.title('Tiempo total vs Tamaño de imagen')
        plt.xlabel('Tamaño (megapixels)')
        plt.ylabel('Tiempo total (ms)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('test_results/time_vs_size.png')
        
        print("\nGráficos guardados en la carpeta 'test_results'")

if __name__ == '__main__':
    unittest.main()
