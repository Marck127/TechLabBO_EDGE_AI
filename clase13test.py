import cv2
import mediapipe as mp
 
RUTA_MODELO = "modelo_lbph_rostro.xml"
TAMANO_ROSTRO = (200, 200)
UMBRAL_CONFIANZA = 65  # ajustar segun pruebas: mas bajo = mas estricto
 
reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.read(RUTA_MODELO)
 
mp_deteccion = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
 
camara = cv2.VideoCapture(0)
 
with mp_deteccion.FaceDetection(model_selection=0, min_detection_confidence=0.6) as deteccion:
    while True:
        ok, frame = camara.read()
        if not ok:
            break
 
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = deteccion.process(rgb)
 
        reconocida = False
 
        if resultado.detections:
            deteccion_rostro = resultado.detections[0]
            caja = deteccion_rostro.location_data.relative_bounding_box
            alto, ancho = frame.shape[:2]
            x1 = max(0, int(caja.xmin * ancho))
            y1 = max(0, int(caja.ymin * alto))
            x2 = min(ancho, int((caja.xmin + caja.width) * ancho))
            y2 = min(alto, int((caja.ymin + caja.height) * alto))
 
            if x2 > x1 and y2 > y1:
                recorte = frame[y1:y2, x1:x2]
                gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
                gris = cv2.resize(gris, TAMANO_ROSTRO)
 
                etiqueta, confianza = reconocedor.predict(gris)
                reconocida = confianza < UMBRAL_CONFIANZA
 
                color = (0, 255, 0) if reconocida else (0, 0, 255)
                texto = f"Reconocida ({confianza:.0f})" if reconocida else f"Desconocida ({confianza:.0f})"
 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, texto, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
        cv2.imshow("Reconocimiento facial", frame)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
camara.release()
cv2.destroyAllWindows()