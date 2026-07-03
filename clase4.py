import cv2
import numpy as np


def cargar_imagen(ruta):

    img = cv2.imread(ruta)

    if img is None:
        raise Exception("No se pudo cargar la imagen")

    return img


# =====================================================
# ELEMENTOS ESTRUCTURANTES
# =====================================================

def elementos_estructurantes():

    rect = cv2.getStructuringElement(
        cv2.MORPH_RECT, (7, 7))

    ellipse = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7))

    cross = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (7, 7))

    print("Rectangular:\n", rect)
    print("\nEliptico:\n", ellipse)
    print("\nCruz:\n", cross)


# =====================================================
# EROSIÓN Y DILATACIÓN
# =====================================================

def erosion_dilatacion(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(binary, kernel, iterations=1)

    dilation = cv2.dilate(binary, kernel, iterations=1)

    cv2.imshow("Original", binary)
    cv2.imshow("Erosion", erosion)
    cv2.imshow("Dilatacion", dilation)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =====================================================
# APERTURA Y CIERRE
# =====================================================

def apertura_cierre(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Original", binary)
    cv2.imshow("Opening", opening)
    cv2.imshow("Closing", closing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =====================================================
# GRADIENTE, TOP-HAT Y BLACK-HAT
# =====================================================

def operaciones_avanzadas(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((9, 9), np.uint8)

    gradient = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, kernel)

    tophat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, kernel)

    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, kernel)

    cv2.imshow("Gradiente", gradient)
    cv2.imshow("TopHat", tophat)
    cv2.imshow("BlackHat", blackhat)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =====================================================
# COMPONENTES CONECTADOS
# =====================================================

def componentes_conectados(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY)

    n_labels, labels = cv2.connectedComponents(binary)

    print("Numero de componentes:", n_labels - 1)

    labels = np.uint8(
        255 * labels / np.max(labels))

    cv2.imshow("Etiquetas", labels)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =====================================================
# CONTORNOS Y DESCRIPTORES
# =====================================================

def contornos_descriptores(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    salida = img.copy()

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)

        x, y, w, h = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)

        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(
                salida, (cx, cy), 5, (0, 0, 255), -1)

        hull = cv2.convexHull(cnt)

        cv2.drawContours(
            salida, [cnt], -1, (0, 255, 0), 2)

        cv2.drawContours(
            salida, [hull], -1, (255, 0, 0), 2)

        cv2.rectangle(
            salida, (x, y),
            (x + w, y + h),
            (255, 255, 0), 2)

        print("Area:", area)
        print("Perimetro:", perimeter)

    cv2.imshow("Contornos", salida)

    cv2.waitKey(0)
    cv2.destroyAllWindows()