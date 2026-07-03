import cv2
import numpy as np


def cargar_imagen(ruta):

    img = cv2.imread(ruta)

    if img is None:
        raise Exception("No se pudo cargar la imagen")

    return img


# ======================================================
# THRESHOLD GLOBAL
# ======================================================

def threshold_global(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow("Original", gray)
    cv2.imshow("Threshold Global", th)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================
# THRESHOLD ADAPTATIVO
# ======================================================

def threshold_adaptativo(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2)

    cv2.imshow("Original", gray)
    cv2.imshow("Adaptativo", th)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================
# OTSU
# ======================================================

def otsu(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("Original", gray)
    cv2.imshow("Otsu", th)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================
# SEGMENTACIÓN HSV
# ======================================================

def segmentacion_hsv(ruta):

    img = cargar_imagen(ruta)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 100, 100])
    upper = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    resultado = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Mascara", mask)
    cv2.imshow("Resultado", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================
# CONNECTED COMPONENTS
# ======================================================

def componentes_conectados(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY)

    n_labels, labels = cv2.connectedComponents(binary)

    print("Numero de objetos:", n_labels - 1)

    labels = np.uint8(
        255 * labels / np.max(labels))

    cv2.imshow("Etiquetas", labels)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================
# SEGMENTACIÓN POR BORDES
# ======================================================

def segmentacion_bordes(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bordes = cv2.Canny(gray, 100, 200)

    cv2.imshow("Bordes", bordes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================
# DISTANCE TRANSFORM
# ======================================================

def distance_transform(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dist = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5)

    dist = cv2.normalize(
        dist,
        None,
        0,
        255,
        cv2.NORM_MINMAX)

    dist = np.uint8(dist)

    cv2.imshow("Distance Transform", dist)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================
# WATERSHED
# ======================================================

def watershed(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5)

    _, sure_fg = cv2.threshold(
        dist,
        0.7 * dist.max(),
        255,
        0)

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    img[markers == -1] = [0, 0, 255]

    cv2.imshow("Watershed", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================================
# K-MEANS
# ======================================================

def kmeans_segmentacion(ruta):

    img = cargar_imagen(ruta)

    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )

    K = 4

    _, labels, centers = cv2.kmeans(
        Z,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    resultado = centers[labels.flatten()]

    resultado = resultado.reshape(img.shape)

    cv2.imshow("KMeans", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()