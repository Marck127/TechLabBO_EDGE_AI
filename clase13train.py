import cv2
import numpy as np
import os
import glob

CARPETA_ROSTROS = "rostros_autorizados"
RUTA_MODELO = "modelo_lbph_rostro.xml"

rutas = glob.glob(os.path.join(CARPETA_ROSTROS, "*.jpg"))

if len(rutas) < 5:
    print(f"Muy pocas fotos ({len(rutas)}). Corre primero clase13capturaref.py")
    exit()

imagenes = []
etiquetas = []

for ruta in rutas:
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if imagen is not None:
        imagenes.append(imagen)
        etiquetas.append(0)  # unica identidad -> etiqueta 0

reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.train(imagenes, np.array(etiquetas))
reconocedor.save(RUTA_MODELO)

print(f"[Entrenamiento] {len(imagenes)} fotos usadas.")
print(f"[Entrenamiento] Modelo guardado en {RUTA_MODELO}")