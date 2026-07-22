# DETECCIÓN DE BOTELLAS CON MOBILE-NET SSD (MODELO DEL PROFESOR)

import cv2
import serial
import time
 
PUERTO_SERIAL = "COM3"
BAUDRATE = 9600
 
RUTA_PROTOTXT = "MobileNetSSD_deploy.prototxt"
RUTA_MODELO = "MobileNetSSD_deploy.caffemodel"
CONFIANZA_MINIMA = 0.5
 
# El modelo fue entrenado en PASCAL VOC (20 clases). "bottle" es la clase 5.
CLASES = ["fondo", "avion", "bicicleta", "pajaro", "bote",
          "botella", "bus", "auto", "gato", "silla",
          "vaca", "mesa", "perro", "caballo", "moto",
          "persona", "planta", "oveja", "sofa", "tren", "monitor"]
INDICE_BOTELLA = CLASES.index("botella")
 
red = cv2.dnn.readNetFromCaffe(RUTA_PROTOTXT, RUTA_MODELO)
 
arduino = serial.Serial(PUERTO_SERIAL, BAUDRATE, timeout=1)
time.sleep(2)
 
camara = cv2.VideoCapture(0)
 
while True:
    ok, frame = camara.read()
    if not ok:
        break
 
    alto, ancho = frame.shape[:2]
 
    # MobileNet-SSD espera imagenes de 300x300, normalizadas
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    red.setInput(blob)
    detecciones = red.forward()
 
    hay_botella = False
 
    for i in range(detecciones.shape[2]):
        confianza = detecciones[0, 0, i, 2]
        if confianza < CONFIANZA_MINIMA:
            continue
 
        clase_id = int(detecciones[0, 0, i, 1])
        if clase_id != INDICE_BOTELLA:
            continue
 
        hay_botella = True
 
        caja = detecciones[0, 0, i, 3:7] * [ancho, alto, ancho, alto]
        x1, y1, x2, y2 = caja.astype(int)
 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Botella {confianza:.0%}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
    arduino.write((b"1\n" if hay_botella else b"0\n"))
 
    texto = "BOTELLA DETECTADA" if hay_botella else "Sin botella"
    color = (0, 255, 0) if hay_botella else (0, 0, 255)
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
 
    cv2.imshow("Deteccion de botellas (modelo profesor)", frame)
 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
camara.release()
cv2.destroyAllWindows()
arduino.close()