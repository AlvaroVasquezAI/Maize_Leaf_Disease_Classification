import unittest
import requests
import os
import time
import shutil
import numpy as np
from PIL import Image, ImageDraw
import io

class TestAPIImageProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://localhost:5000"
        # Prepara carpeta temporal
        cls.tmp_dir = 'temp_test_images'
        if os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)
        os.makedirs(cls.tmp_dir, exist_ok=True)

        # Buscar imágenes reales en data/test
        cls.real_images = []
        for class_dir in sorted(os.listdir('data/test')):
            full_dir = os.path.join('data/test', class_dir)
            if not os.path.isdir(full_dir):
                continue
            imgs = [f for f in os.listdir(full_dir)
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff','.tif','.webp'))]
            # toma sólo la primera (o las dos primeras) de cada clase
            for img in imgs[:2]:
                cls.real_images.append(os.path.join(full_dir, img))

        # Si no hay reales, generar sintéticas
        if not cls.real_images:
            print("No se encontraron imágenes reales, generando sintéticas...")
            img = Image.new('RGB', (224, 224), color='green')
            draw = ImageDraw.Draw(img)
            draw.ellipse((50,50,175,175), fill='brown')
            path = os.path.join(cls.tmp_dir, 'synthetic.jpg')
            img.save(path)
            cls.real_images = [path]

        # Convertir cada real a varios formatos
        cls.test_images = []  # list of tuples (label, path, mime)
        formats = [('.jpg', 'image/jpeg'), ('.png', 'image/png'), ('.bmp', 'image/bmp'), ('.tiff', 'image/tiff')]
        for img_path in cls.real_images:
            base = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(img_path)
            for ext, mime in formats:
                out_path = os.path.join(cls.tmp_dir, f"{base}{ext}")
                img.save(out_path)
                cls.test_images.append((f"{base}{ext}", out_path, mime))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_different_image_sizes_and_formats(self):
        """Verificar procesamiento de múltiples tamaños y formatos reales."""
        results = []
        for label, img_path, mime in self.test_images:
            with Image.open(img_path) as img:
                size = img.size

            with open(img_path, 'rb') as f:
                files = {'image': (os.path.basename(img_path), f, mime)}
                start = time.time()
                resp = requests.post(f"{self.base_url}/predict", files=files)
                elapsed = time.time() - start

            self.assertEqual(resp.status_code, 200, f"{label} fallo con {resp.status_code}")
            data = resp.json()
            self.assertIn('className', data)
            self.assertIn('confidence', data)
            self.assertIn('processingTime', data)

            results.append((label, size, elapsed, data['processingTime'], data['className'], data['confidence']))

        # Mostrar resumen
        print("\nResumen de pruebas de tamaños y formatos reales:")
        print(f"{'Imagen':<25}{'Size':<15}{'RT (ms)':<10}{'Srv (ms)':<10}{'Clase':<20}{'Conf':<8}")
        print('-'*88)
        for lbl, sz, rt, st, cls_nm, conf in results:
            print(f"{lbl:<25}{str(sz):<15}{rt*1000:<10.1f}{st*1000:<10.1f}{cls_nm:<20}{conf:.4f}")

    def test_image_format_validation(self):
        """Verificar aceptación de formatos BMP y TIFF además de JPG/PNG."""
        for label, img_path, mime in self.test_images:
            with open(img_path, 'rb') as f:
                files = {'image': (os.path.basename(img_path), f, mime)}
                resp = requests.post(f"{self.base_url}/predict", files=files)
            # Los formatos válidos deben devolver 200
            self.assertEqual(resp.status_code, 200, f"Formato {mime} rechazado: {resp.status_code}")
        # Probar formato inválido
        txt = os.path.join(self.tmp_dir, 'invalid.txt')
        with open(txt, 'w') as f: f.write('not an image')
        with open(txt, 'rb') as f:
            files = {'image': ('invalid.txt', f, 'text/plain')}
            resp = requests.post(f"{self.base_url}/predict", files=files)
        self.assertNotEqual(resp.status_code, 200)

    def test_server_processing_time_limits(self):
        """Garantizar que processingTime está en rango razonable (<10s)."""
        label, img_path, mime = self.test_images[0]
        with open(img_path, 'rb') as f:
            files = {'image': (os.path.basename(img_path), f, mime)}
            resp = requests.post(f"{self.base_url}/predict", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        pt = data['processingTime']
        self.assertGreater(pt, 0)
        self.assertLess(pt, 10)

if __name__ == '__main__':
    unittest.main()
