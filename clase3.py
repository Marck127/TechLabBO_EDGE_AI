import cv2
import numpy as np


# ==========================================================
# FUNCIÓN AUXILIAR
# ==========================================================

def cargar_imagen(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        raise Exception(f"No se pudo cargar la imagen: {ruta}")

    return imagen


# ==========================================================
# ESPACIOS DE COLOR
# ==========================================================

def espacios_color(ruta):

    img = cargar_imagen(ruta)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    cv2.imshow("BGR", img)
    cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("HSV", hsv)
    cv2.imshow("LAB", lab)
    cv2.imshow("YUV", yuv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# DETECCIÓN DE COLOR EN HSV
# ==========================================================

def deteccion_color(ruta):

    img = cargar_imagen(ruta)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rango para color rojo
    lower = np.array([0, 120, 70])
    upper = np.array([10, 255, 255])

    mascara = cv2.inRange(hsv, lower, upper)

    resultado = cv2.bitwise_and(img, img, mask=mascara)

    cv2.imshow("Original", img)
    cv2.imshow("Mascara Roja", mascara)
    cv2.imshow("Resultado", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# HISTOGRAMA
# ==========================================================

def histograma(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)

    cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)

    for x, y in enumerate(hist):
        cv2.line(hist_img,
                 (x, 300),
                 (x, 300 - int(y)),
                 (255, 255, 255), 1)

    cv2.imshow("Imagen", gray)
    cv2.imshow("Histograma", hist_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# ECUALIZACIÓN
# ==========================================================

def ecualizacion(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equalizada = cv2.equalizeHist(gray)

    cv2.imshow("Original", gray)
    cv2.imshow("Ecualizada", equalizada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# CLAHE
# ==========================================================

def clahe(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    resultado = clahe.apply(gray)

    cv2.imshow("Original", gray)
    cv2.imshow("CLAHE", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# CONTRAST STRETCHING
# ==========================================================

def contrast_stretching(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    minimo = np.min(gray)
    maximo = np.max(gray)

    stretched = ((gray - minimo) /
                 (maximo - minimo) * 255)

    stretched = stretched.astype(np.uint8)

    cv2.imshow("Original", gray)
    cv2.imshow("Contrast Stretching", stretched)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# FILTROS
# ==========================================================

def filtros(ruta):

    img = cargar_imagen(ruta)

    box = cv2.blur(img, (7, 7))

    gauss = cv2.GaussianBlur(img, (7, 7), 0)

    median = cv2.medianBlur(img, 7)

    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    cv2.imshow("Original", img)
    cv2.imshow("Box Filter", box)
    cv2.imshow("Gaussian", gauss)
    cv2.imshow("Median", median)
    cv2.imshow("Bilateral", bilateral)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# SHARPEN Y EMBOSS
# ==========================================================

def realce(ruta):

    img = cargar_imagen(ruta)

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])

    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    emboss = cv2.filter2D(img, -1, emboss_kernel)

    cv2.imshow("Original", img)
    cv2.imshow("Sharpen", sharpen)
    cv2.imshow("Emboss", emboss)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# SOBEL
# ==========================================================

def sobel(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel_xy = cv2.addWeighted(
        sobel_x, 0.5,
        sobel_y, 0.5,
        0
    )

    cv2.imshow("Sobel X", sobel_x)
    cv2.imshow("Sobel Y", sobel_y)
    cv2.imshow("Sobel XY", sobel_xy)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# LAPLACIANO
# ==========================================================

def laplaciano(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    lap = cv2.convertScaleAbs(lap)

    cv2.imshow("Original", gray)
    cv2.imshow("Laplaciano", lap)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# CANNY
# ==========================================================

def canny(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bordes = cv2.Canny(gray, 100, 200)

    cv2.imshow("Original", gray)
    cv2.imshow("Canny", bordes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# TRANSFORMADA DE HOUGH
# ==========================================================

def hough_lineas(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bordes = cv2.Canny(gray, 100, 200)

    lineas = cv2.HoughLinesP(
        bordes,
        1,
        np.pi / 180,
        100,
        minLineLength=50,
        maxLineGap=10
    )

    if lineas is not None:

        for linea in lineas:

            x1, y1, x2, y2 = linea[0]

            cv2.line(img,
                     (x1, y1),
                     (x2, y2),
                     (0, 255, 0),
                     2)

    cv2.imshow("Hough Lineas", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# HOUGH CÍRCULOS
# ==========================================================

def hough_circulos(ruta):

    img = cargar_imagen(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=150
    )

    if circles is not None:

        circles = np.uint16(np.around(circles))

        for c in circles[0, :]:

            cv2.circle(img,
                       (c[0], c[1]),
                       c[2],
                       (0, 255, 0),
                       2)

    cv2.imshow("Hough Circulos", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()