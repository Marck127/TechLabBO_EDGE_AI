import cv2
import numpy as np
import serial
import time

PUERTO_SERIAL = "COM4"
BAUDRATE = 9600
UMBRAL_NEGRO = 70
AREA_MINIMA_CUADRO = 4000
TAMANO_BUFFER = 8
KERNEL_MORFOLOGICO = np.ones((5, 5), np.uint8)

COLOR_BGR = {"R": (0, 0, 255), "G": (0, 255, 0), "B": (255, 0, 0), "N": (200, 200, 200)}
TEXTO_COLOR = {"R": "ROJO", "G": "VERDE", "B": "AZUL", "N": "NINGUNO"}

RANGO_ROJO_1 = ((0, 100, 70), (10, 255, 255))
RANGO_ROJO_2 = ((170, 100, 70), (180, 255, 255))
RANGO_VERDE = ((40, 70, 60), (85, 255, 255))
RANGO_AZUL = ((90, 50, 40), (135, 255, 255))

arduino = serial.Serial(PUERTO_SERIAL, BAUDRATE, timeout=1)
time.sleep(2)

camara = cv2.VideoCapture(0)
buffer_colores = []
buffer_fps = []
TAMANO_BUFFER_FPS = 15
ultimo_color_enviado = None
tiempo_anterior = time.time()

while True:
    ok, frame = camara.read()
    if not ok:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, UMBRAL_NEGRO, 255, cv2.THRESH_BINARY_INV)

    contornos, jerarquia = cv2.findContours(binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cuadro_encontrado = None
    interior_encontrado = None

    if jerarquia is not None:
        for i, contorno in enumerate(contornos):
            area = cv2.contourArea(contorno)
            if area < AREA_MINIMA_CUADRO:
                continue

            perimetro = cv2.arcLength(contorno, True)
            aprox = cv2.approxPolyDP(contorno, 0.03 * perimetro, True)
            if len(aprox) != 4:
                continue

            hijo = jerarquia[0][i][2]
            if hijo == -1:
                continue

            cuadro_encontrado = aprox
            interior_encontrado = contornos[hijo]
            break

    color_detectado = "N"

    if cuadro_encontrado is not None:
        cv2.drawContours(frame, [cuadro_encontrado], -1, (0, 255, 255), 2)

        x, y, w, h = cv2.boundingRect(interior_encontrado)
        roi = frame[y:y + h, x:x + w]

        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mascara_color = cv2.inRange(hsv, (0, 80, 40), (180, 255, 255))
            mascara_color = cv2.morphologyEx(mascara_color, cv2.MORPH_OPEN, KERNEL_MORFOLOGICO)
            mascara_color = cv2.morphologyEx(mascara_color, cv2.MORPH_CLOSE, KERNEL_MORFOLOGICO)
            contornos_circulo, _ = cv2.findContours(mascara_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mejor_circulo = None
            mejor_circularidad = 0

            for c in contornos_circulo:
                area = cv2.contourArea(c)
                if area < 150:
                    continue
                perimetro = cv2.arcLength(c, True)
                if perimetro == 0:
                    continue
                circularidad = 4 * np.pi * area / (perimetro * perimetro)
                if circularidad > mejor_circularidad:
                    mejor_circularidad = circularidad
                    mejor_circulo = c

            if mejor_circulo is not None and mejor_circularidad > 0.7:
                mascara_circulo = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.drawContours(mascara_circulo, [mejor_circulo], -1, 255, -1)

                mascara_r_c = cv2.bitwise_and(
                    cv2.inRange(hsv, RANGO_ROJO_1[0], RANGO_ROJO_1[1]) |
                    cv2.inRange(hsv, RANGO_ROJO_2[0], RANGO_ROJO_2[1]),
                    mascara_circulo)
                mascara_g_c = cv2.bitwise_and(cv2.inRange(hsv, RANGO_VERDE[0], RANGO_VERDE[1]), mascara_circulo)
                mascara_b_c = cv2.bitwise_and(cv2.inRange(hsv, RANGO_AZUL[0], RANGO_AZUL[1]), mascara_circulo)

                conteos = {
                    "R": cv2.countNonZero(mascara_r_c),
                    "G": cv2.countNonZero(mascara_g_c),
                    "B": cv2.countNonZero(mascara_b_c),
                }
                color_ganador, conteo_ganador = max(conteos.items(), key=lambda kv: kv[1])
                area_circulo = cv2.countNonZero(mascara_circulo)

                if area_circulo > 0 and conteo_ganador > 0.3 * area_circulo:
                    color_detectado = color_ganador

    buffer_colores.append(color_detectado)
    if len(buffer_colores) > TAMANO_BUFFER:
        buffer_colores.pop(0)

    color_estable = max(set(buffer_colores), key=buffer_colores.count)

    if color_estable != ultimo_color_enviado:
        ultimo_color_enviado = color_estable
        arduino.write((color_estable + "\n").encode())
        print(f"Color enviado: {color_estable}")

    tiempo_actual = time.time()
    fps_instantaneo = 1.0 / max(tiempo_actual - tiempo_anterior, 1e-6)
    tiempo_anterior = tiempo_actual

    buffer_fps.append(fps_instantaneo)
    if len(buffer_fps) > TAMANO_BUFFER_FPS:
        buffer_fps.pop(0)
    fps = sum(buffer_fps) / len(buffer_fps)

    cv2.putText(frame, f"Color: {TEXTO_COLOR[color_estable]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BGR[color_estable], 2)
    cv2.putText(frame, f"Enviado por serial: {color_estable}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BGR[color_estable], 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Camara", frame)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mascara_r = cv2.inRange(hsv_frame, RANGO_ROJO_1[0], RANGO_ROJO_1[1]) | \
                cv2.inRange(hsv_frame, RANGO_ROJO_2[0], RANGO_ROJO_2[1])
    mascara_g = cv2.inRange(hsv_frame, RANGO_VERDE[0], RANGO_VERDE[1])
    mascara_b = cv2.inRange(hsv_frame, RANGO_AZUL[0], RANGO_AZUL[1])

    mascara_r = cv2.morphologyEx(mascara_r, cv2.MORPH_OPEN, KERNEL_MORFOLOGICO)
    mascara_g = cv2.morphologyEx(mascara_g, cv2.MORPH_OPEN, KERNEL_MORFOLOGICO)
    mascara_b = cv2.morphologyEx(mascara_b, cv2.MORPH_OPEN, KERNEL_MORFOLOGICO)

    salida_r = np.zeros_like(frame)
    salida_r[:, :, 2] = mascara_r

    salida_g = np.zeros_like(frame)
    salida_g[:, :, 1] = mascara_g

    salida_b = np.zeros_like(frame)
    salida_b[:, :, 0] = mascara_b

    cv2.imshow("Mascara ROJO", salida_r)
    cv2.imshow("Mascara VERDE", salida_g)
    cv2.imshow("Mascara AZUL", salida_b)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()
arduino.close()