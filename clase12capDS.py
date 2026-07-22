# Capturando imágenes para el dataset de botellas y sin botellas - Model Student

import cv2
import os
 
CARPETA_BASE = "dataset_botellas"
CARPETA_POSITIVA = os.path.join(CARPETA_BASE, "botella")
CARPETA_NEGATIVA = os.path.join(CARPETA_BASE, "sin_botella")
 
os.makedirs(CARPETA_POSITIVA, exist_ok=True)
os.makedirs(CARPETA_NEGATIVA, exist_ok=True)
 
contador_pos = len(os.listdir(CARPETA_POSITIVA))
contador_neg = len(os.listdir(CARPETA_NEGATIVA))
 
camara = cv2.VideoCapture(0)
 
print("Presiona 'b' = guardar como BOTELLA, 'n' = guardar como SIN BOTELLA, 'q' = salir")
 
while True:
    ok, frame = camara.read()
    if not ok:
        break
 
    cv2.putText(frame, f"Botella: {contador_pos}  Sin botella: {contador_neg}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Captura de dataset", frame)
 
    tecla = cv2.waitKey(1) & 0xFF
 
    if tecla == ord("b"):
        ruta = os.path.join(CARPETA_POSITIVA, f"botella_{contador_pos:03d}.jpg")
        cv2.imwrite(ruta, frame)
        contador_pos += 1
        print(f"Guardada: {ruta}")
 
    elif tecla == ord("n"):
        ruta = os.path.join(CARPETA_NEGATIVA, f"sin_botella_{contador_neg:03d}.jpg")
        cv2.imwrite(ruta, frame)
        contador_neg += 1
        print(f"Guardada: {ruta}")
 
    elif tecla == ord("q"):
        break
 
camara.release()
cv2.destroyAllWindows()