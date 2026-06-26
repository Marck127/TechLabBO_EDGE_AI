import cv2
import numpy as np


# ==========================================================
# CARGAR IMÁGENES
# ==========================================================

def cargar_imagenes(path):

    img_bgr = cv2.imread(path)

    if img_bgr is None:
        raise Exception(f"No se pudo abrir la imagen: {path}")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_bgr, img_gray, img_rgb


# ==========================================================
# SUMA
# ==========================================================

def suma(path):

    _, img_gray, _ = cargar_imagenes(path)

    resultado = cv2.add(img_gray, 50)

    cv2.imshow("Original", img_gray)
    cv2.imshow("Suma +50", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# RESTA
# ==========================================================

def resta(path):

    _, img_gray, _ = cargar_imagenes(path)

    resultado = cv2.subtract(img_gray, 50)

    cv2.imshow("Original", img_gray)
    cv2.imshow("Resta -50", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# MULTIPLICACIÓN POR ESCALAR
# ==========================================================

def multiplicacion(path):

    _, img_gray, _ = cargar_imagenes(path)

    resultado = cv2.convertScaleAbs(
        img_gray,
        alpha=1.5,
        beta=0
    )

    cv2.imshow("Original", img_gray)
    cv2.imshow("Multiplicacion x1.5", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# OPERACIONES LÓGICAS
# ==========================================================

def operaciones_logicas(path):

    _, img_gray, _ = cargar_imagenes(path)

    mask = np.zeros_like(img_gray)

    cv2.circle(
        mask,
        (img_gray.shape[1] // 2,
         img_gray.shape[0] // 2),
        100,
        255,
        -1
    )

    img_and = cv2.bitwise_and(img_gray, mask)
    img_or = cv2.bitwise_or(img_gray, mask)
    img_xor = cv2.bitwise_xor(img_gray, mask)
    img_not = cv2.bitwise_not(img_gray)

    cv2.imshow("Mascara", mask)
    cv2.imshow("AND", img_and)
    cv2.imshow("OR", img_or)
    cv2.imshow("XOR", img_xor)
    cv2.imshow("NOT", img_not)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# TRASLACIÓN
# ==========================================================

def traslacion(path):

    img_bgr, _, _ = cargar_imagenes(path)

    alto, ancho = img_bgr.shape[:2]

    tx = 100
    ty = 50

    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    resultado = cv2.warpAffine(
        img_bgr,
        M,
        (ancho, alto)
    )

    cv2.imshow("Original", img_bgr)
    cv2.imshow("Traslacion", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# ROTACIÓN
# ==========================================================

def rotacion(path):

    img_bgr, _, _ = cargar_imagenes(path)

    alto, ancho = img_bgr.shape[:2]

    centro = (ancho // 2, alto // 2)

    M = cv2.getRotationMatrix2D(
        centro,
        45,
        1.0
    )

    resultado = cv2.warpAffine(
        img_bgr,
        M,
        (ancho, alto)
    )

    cv2.imshow("Original", img_bgr)
    cv2.imshow("Rotacion", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# ESCALADO
# ==========================================================

def escalado(path):

    img_bgr, _, _ = cargar_imagenes(path)

    resultado = cv2.resize(
        img_bgr,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imshow("Original", img_bgr)
    cv2.imshow("Escalado", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# REFLEXIÓN
# ==========================================================

def reflexion(path):

    img_bgr, _, _ = cargar_imagenes(path)

    horizontal = cv2.flip(img_bgr, 1)
    vertical = cv2.flip(img_bgr, 0)
    ambas = cv2.flip(img_bgr, -1)

    cv2.imshow("Original", img_bgr)
    cv2.imshow("Horizontal", horizontal)
    cv2.imshow("Vertical", vertical)
    cv2.imshow("Ambas", ambas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()