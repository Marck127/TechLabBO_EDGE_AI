import cv2


# ==========================================================
# ABRIR IMAGEN
# ==========================================================

def abrir_imagen(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        raise Exception(f"No se pudo abrir la imagen: {ruta}")

    return imagen


# ==========================================================
# MOSTRAR IMAGEN EN BGR
# ==========================================================

def mostrar_bgr(ruta):

    imagen = abrir_imagen(ruta)

    cv2.imshow("Imagen BGR", imagen)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# MOSTRAR IMAGEN EN ESCALA DE GRISES
# ==========================================================

def mostrar_grises(ruta):

    imagen = abrir_imagen(ruta)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Imagen Escala de Grises", gris)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# MOSTRAR IMAGEN EN RGB
# ==========================================================

def mostrar_rgb(ruta):

    imagen = abrir_imagen(ruta)

    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # OpenCV muestra correctamente en BGR,
    # por ello reconvertimos antes de visualizar
    rgb_visualizacion = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("Imagen RGB", rgb_visualizacion)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# MOSTRAR TODAS LAS VERSIONES
# ==========================================================

def mostrar_todas(ruta):

    imagen = abrir_imagen(ruta)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    rgb_visualizacion = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("BGR", imagen)
    cv2.imshow("Escala de Grises", gris)
    cv2.imshow("RGB", rgb_visualizacion)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# ABRIR CÁMARA WEB
# ==========================================================

def abrir_camara():

    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        raise Exception("No se pudo abrir la cámara")

    print("Presione 'q' para salir")

    while True:

        ret, frame = camara.read()

        if not ret:
            break

        cv2.imshow("Camara Web", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q'):
            break

    camara.release()
    cv2.destroyAllWindows()