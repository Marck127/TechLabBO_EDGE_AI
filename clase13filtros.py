import cv2
import numpy as np
import mediapipe as mp
import time
import serial
from websockets.sync.client import connect as conectar_websocket

PUERTO_SERIAL = "COM3"
BAUDRATE = 9600
IP_ESP32 = "192.168.1.100"

RUTA_MODELO = "modelo_lbph_rostro.xml"
TAMANO_ROSTRO = (200, 200)
UMBRAL_CONFIANZA = 65

mp_deteccion = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Indices de landmarks de Face Mesh que usamos para posicionar los filtros
LM_NARIZ = 1
LM_FRENTE = 10
LM_OREJA_IZQ = 234
LM_OREJA_DER = 454
LM_MANDIBULA = [58, 172, 136, 150, 176, 148, 152, 377, 400, 379, 365, 397, 288]

filtro_actual = "ninguno"  # "ninguno" | "perrito" | "barba"


def punto_px(landmark, ancho, alto):
    return int(landmark.x * ancho), int(landmark.y * alto)


def dibujar_filtro_perrito(frame, landmarks, ancho, alto):
    overlay = frame.copy()

    nariz = punto_px(landmarks[LM_NARIZ], ancho, alto)
    oreja_izq = punto_px(landmarks[LM_OREJA_IZQ], ancho, alto)
    oreja_der = punto_px(landmarks[LM_OREJA_DER], ancho, alto)

    # Nariz de perrito
    cv2.circle(overlay, nariz, 18, (20, 20, 20), -1)
    cv2.circle(overlay, (nariz[0] - 6, nariz[1] - 5), 4, (255, 255, 255), -1)

    # Orejas caidas (elipses cafe a los lados de la frente)
    cv2.ellipse(overlay, (oreja_izq[0] - 15, oreja_izq[1] - 35), (28, 48), -25, 0, 360, (19, 69, 139), -1)
    cv2.ellipse(overlay, (oreja_der[0] + 15, oreja_der[1] - 35), (28, 48), 25, 0, 360, (19, 69, 139), -1)

    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


def dibujar_filtro_barba(frame, landmarks, ancho, alto):
    overlay = frame.copy()

    puntos = np.array([punto_px(landmarks[i], ancho, alto) for i in LM_MANDIBULA], dtype=np.int32)
    cv2.fillPoly(overlay, [puntos], (15, 15, 15))

    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)


def detectar_y_enviar(enviar_comando):
    global filtro_actual

    reconocedor = cv2.face.LBPHFaceRecognizer_create()
    reconocedor.read(RUTA_MODELO)

    camara = cv2.VideoCapture(0)
    ultimo_comando = None

    with mp_deteccion.FaceDetection(model_selection=0, min_detection_confidence=0.6) as deteccion, \
         mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6,
                                min_tracking_confidence=0.6) as face_mesh:

        while True:
            ok, frame = camara.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resultado_deteccion = deteccion.process(rgb)
            reconocida = False

            if resultado_deteccion.detections:
                caja = resultado_deteccion.detections[0].location_data.relative_bounding_box
                x1 = max(0, int(caja.xmin * ancho))
                y1 = max(0, int(caja.ymin * alto))
                x2 = min(ancho, int((caja.xmin + caja.width) * ancho))
                y2 = min(alto, int((caja.ymin + caja.height) * alto))

                if x2 > x1 and y2 > y1:
                    recorte = frame[y1:y2, x1:x2]
                    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
                    gris = cv2.resize(gris, TAMANO_ROSTRO)

                    _, confianza = reconocedor.predict(gris)
                    reconocida = confianza < UMBRAL_CONFIANZA

                    color = (0, 255, 0) if reconocida else (0, 0, 255)
                    texto = "Reconocida" if reconocida else "Desconocida"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, texto, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # El filtro SOLO se aplica si la persona fue reconocida
                    if reconocida and filtro_actual != "ninguno":
                        resultado_malla = face_mesh.process(rgb)
                        if resultado_malla.multi_face_landmarks:
                            landmarks = resultado_malla.multi_face_landmarks[0].landmark
                            if filtro_actual == "perrito":
                                dibujar_filtro_perrito(frame, landmarks, ancho, alto)
                            elif filtro_actual == "barba":
                                dibujar_filtro_barba(frame, landmarks, ancho, alto)

            comando = "1" if reconocida else "0"
            if comando != ultimo_comando:
                ultimo_comando = comando
                enviar_comando(comando)
                print(f"Enviado: {comando}")

            cv2.putText(frame, f"Filtro: {filtro_actual} (1/2/3 para cambiar)", (10, alto - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Reconocimiento + Filtros", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("1"):
                filtro_actual = "ninguno"
            elif tecla == ord("2"):
                filtro_actual = "perrito"
            elif tecla == ord("3"):
                filtro_actual = "barba"

    camara.release()
    cv2.destroyAllWindows()


def arduino_Ser():
    """Conecta por cable serial al Arduino/ESP32."""
    arduino = serial.Serial(PUERTO_SERIAL, BAUDRATE, timeout=1)
    time.sleep(2)

    def enviar_comando(comando):
        arduino.write((comando + "\n").encode())

    detectar_y_enviar(enviar_comando)
    arduino.close()


def esp32_WS():
    """Conecta por WebSocket al ESP32 (misma WiFi)."""
    with conectar_websocket(f"ws://{IP_ESP32}:81/") as ws:
        def enviar_comando(comando):
            ws.send(comando)

        detectar_y_enviar(enviar_comando)


if __name__ == "__main__":
    arduino_Ser()
    # esp32_WS()