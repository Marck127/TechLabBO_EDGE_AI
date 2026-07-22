import cv2
import serial
import time
 
PUERTO_SERIAL = "COM3"
BAUDRATE = 9600
TAMANO_ESTUDIANTE = 6
 
arduino = serial.Serial(PUERTO_SERIAL, BAUDRATE, timeout=1)
time.sleep(2)
 
camara = cv2.VideoCapture(0)
 
while True:
    ok, frame = camara.read()
    if not ok:
        break
 
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    chica = cv2.resize(gris, (TAMANO_ESTUDIANTE, TAMANO_ESTUDIANTE))
    vector = (chica.flatten().astype(float) / 255.0)
 
    comando = ",".join(f"{v:.4f}" for v in vector)
    arduino.write((comando + "\n").encode())
 
    # Muestra tambien la "vision" comprimida que recibe el Arduino,
    # para que en clase se vea claramente que tan poco ve el estudiante.
    vista_comprimida = cv2.resize(chica, (240, 240), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Lo que ve el ESTUDIANTE (6x6 real)", vista_comprimida)
    cv2.imshow("Camara", frame)
 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
camara.release()
cv2.destroyAllWindows()
arduino.close()