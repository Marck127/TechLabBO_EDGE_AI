import cv2
import mediapipe as mp
import os

CARPETA_SALIDA = "rostros_autorizados"
TAMANO_ROSTRO = (200, 200)

os.makedirs(CARPETA_SALIDA, exist_ok=True)
contador = len(os.listdir(CARPETA_SALIDA))

mp_deteccion = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

camara = cv2.VideoCapture(0)

print("Presiona 'c' para capturar, 'q' para salir.")

with mp_deteccion.FaceDetection(model_selection=0, min_detection_confidence=0.6) as deteccion:
    while True:
        ok, frame = camara.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = deteccion.process(rgb)

        recorte_actual = None

        if resultado.detections:
            deteccion_rostro = resultado.detections[0]
            mp_draw.draw_detection(frame, deteccion_rostro)

            caja = deteccion_rostro.location_data.relative_bounding_box
            alto, ancho = frame.shape[:2]
            x1 = max(0, int(caja.xmin * ancho))
            y1 = max(0, int(caja.ymin * alto))
            x2 = min(ancho, int((caja.xmin + caja.width) * ancho))
            y2 = min(alto, int((caja.ymin + caja.height) * alto))

            if x2 > x1 and y2 > y1:
                recorte = frame[y1:y2, x1:x2]
                gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
                recorte_actual = cv2.resize(gris, TAMANO_ROSTRO)

        cv2.putText(frame, f"Capturadas: {contador}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Captura de rostro autorizado", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("c") and recorte_actual is not None:
            ruta = os.path.join(CARPETA_SALIDA, f"rostro_{contador:03d}.jpg")
            cv2.imwrite(ruta, recorte_actual)
            contador += 1
            print(f"Guardada: {ruta}")
        elif tecla == ord("q"):
            break

camara.release()
cv2.destroyAllWindows()