# MAIN.PY
# ==========================================================
# IMPORTAR MÓDULOS
# ==========================================================

from clase1 import *
from clase2 import *
from clase3 import *


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================

def main():

    # Ruta de la imagen, receurda que éste puede cambiar según la ubicación de tu imagen
    ruta = "img1.jpg"

    # ======================================================
    # CLASE 1
    # ======================================================

    print("\n=== CLASE 1 ===")

    # Mostrar imagen en BGR
    mostrar_bgr(ruta)

    # Mostrar imagen en escala de grises
    mostrar_grises(ruta)

    # Mostrar imagen en RGB
    mostrar_rgb(ruta)

    # Mostrar todas simultáneamente
    mostrar_todas(ruta)

    # Abrir cámara (descomentar si se desea usar)
    # abrir_camara()


    # ======================================================
    # CLASE 2 - OPERACIONES BÁSICAS
    # ======================================================

    print("\n=== CLASE 2: OPERACIONES BÁSICAS ===")

    suma(ruta)

    resta(ruta)

    multiplicacion(ruta)

    operaciones_logicas(ruta)


    # ======================================================
    # CLASE 2 - TRANSFORMACIONES GEOMÉTRICAS
    # ======================================================

    print("\n=== CLASE 2: TRANSFORMACIONES GEOMÉTRICAS ===")

    traslacion(ruta)

    rotacion(ruta)

    escalado(ruta)

    reflexion(ruta)

    # ======================================================
    # CLASE 3 - PRE-PROCESAMIENTO DE IMÁGENES
    # ======================================================

    print("\n=== CLASE 3: PRE-PROCESAMIENTO DE IMÁGENES ===")

    espacios_color(ruta)

    deteccion_color(ruta)

    histograma(ruta)

    ecualizacion(ruta)

    clahe(ruta)

    contrast_stretching(ruta)

    filtros(ruta)

    realce(ruta)

    sobel(ruta)

    laplaciano(ruta)

    canny(ruta)

    hough_lineas(ruta)

    hough_circulos(ruta)

# ==========================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# ==========================================================

if __name__ == "__main__":
    main()
