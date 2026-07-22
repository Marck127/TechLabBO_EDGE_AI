# Entrenamiento de un modelo diminuto (student) para detectar botellas

import os
import glob
import numpy as np
import cv2
 
CARPETA_DATASET = "dataset_botellas"
RUTA_PROTOTXT = "MobileNetSSD_deploy.prototxt"
RUTA_MODELO = "MobileNetSSD_deploy.caffemodel"
 
TAMANO_ESTUDIANTE = 6          # imagen comprimida a 6x6 para el estudiante
NEURONAS_OCULTAS = 8
ALPHA_DESTILACION = 0.7        # 0.7 = confiar mas en el profesor, 0.3 en la etiqueta real
TASA_APRENDIZAJE = 0.5
EPOCAS = 200
 
INDICE_BOTELLA = 5  # clase "bottle" en PASCAL VOC (ver deteccion_botellas_profesor.py)
 
def confianza_del_profesor(red, imagen_bgr):
    """Corre el profesor y devuelve la confianza maxima de 'botella' (0.0 si no ve ninguna)."""
    alto, ancho = imagen_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(imagen_bgr, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    red.setInput(blob)
    detecciones = red.forward()
 
    mejor_confianza = 0.0
    for i in range(detecciones.shape[2]):
        clase_id = int(detecciones[0, 0, i, 1])
        confianza = float(detecciones[0, 0, i, 2])
        if clase_id == INDICE_BOTELLA and confianza > mejor_confianza:
            mejor_confianza = confianza
    return mejor_confianza
 
 
def comprimir_para_estudiante(imagen_bgr):
    """Reduce la imagen a NxN en escala de grises, normalizada 0-1 -> lo que 'veria' el MCU."""
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    chica = cv2.resize(gris, (TAMANO_ESTUDIANTE, TAMANO_ESTUDIANTE))
    return (chica.flatten().astype(np.float32) / 255.0)
 
 
def construir_dataset_destilacion():
    print("[Profesor] Cargando MobileNet-SSD...")
    red_profesor = cv2.dnn.readNetFromCaffe(RUTA_PROTOTXT, RUTA_MODELO)
 
    X, y_suave, y_duro = [], [], []
 
    for etiqueta_dura, subcarpeta in [(1.0, "botella"), (0.0, "sin_botella")]:
        rutas = glob.glob(os.path.join(CARPETA_DATASET, subcarpeta, "*.jpg"))
        print(f"[Datos] {subcarpeta}: {len(rutas)} imagenes")
 
        for ruta in rutas:
            imagen = cv2.imread(ruta)
            if imagen is None:
                continue
 
            confianza = confianza_del_profesor(red_profesor, imagen)
            X.append(comprimir_para_estudiante(imagen))
            y_suave.append(confianza)
            y_duro.append(etiqueta_dura)
 
    return np.array(X), np.array(y_suave), np.array(y_duro)
 
 
# ==============================
# Red diminuta (numpy puro, sin frameworks pesados)
# ==============================
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))
 
 
def entrenar_estudiante(X, y_objetivo):
    n_entradas = X.shape[1]
    rng = np.random.default_rng(42)
 
    W1 = rng.normal(0, 0.3, (n_entradas, NEURONAS_OCULTAS))
    b1 = np.zeros(NEURONAS_OCULTAS)
    W2 = rng.normal(0, 0.3, (NEURONAS_OCULTAS, 1))
    b2 = np.zeros(1)
 
    n = X.shape[0]
 
    for epoca in range(EPOCAS):
        # --- forward ---
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)          # ReLU
        z2 = a1 @ W2 + b2
        a2 = sigmoide(z2).flatten()     # salida final (0-1)
 
        # --- perdida (solo para mostrar progreso) ---
        perdida = np.mean((a2 - y_objetivo) ** 2)
 
        # --- backward (gradiente de MSE + sigmoide + ReLU) ---
        d_a2 = (2 / n) * (a2 - y_objetivo)
        d_z2 = d_a2 * a2 * (1 - a2)
        d_W2 = a1.T @ d_z2.reshape(-1, 1)
        d_b2 = d_z2.sum()
 
        d_a1 = d_z2.reshape(-1, 1) @ W2.T
        d_z1 = d_a1 * (z1 > 0)
        d_W1 = X.T @ d_z1
        d_b1 = d_z1.sum(axis=0)
 
        W1 -= TASA_APRENDIZAJE * d_W1
        b1 -= TASA_APRENDIZAJE * d_b1
        W2 -= TASA_APRENDIZAJE * d_W2
        b2 -= TASA_APRENDIZAJE * d_b2
 
        if epoca % 50 == 0:
            print(f"  epoca {epoca:4d}  perdida={perdida:.4f}")
 
    return W1, b1, W2, b2
 
 
def exportar_a_c(W1, b1, W2, b2, ruta_salida="estudiante_botella_generado.h"):
    def formatear_array(nombre, arr):
        valores = ", ".join(f"{v:.6f}f" for v in arr.flatten())
        return f"const float {nombre}[{arr.size}] = {{{valores}}};"
 
    lineas = [
        "// Generado automaticamente por entrenar_estudiante_botella.py",
        "// Red: 36 entradas (imagen 6x6 gris) -> 8 ocultas (ReLU) -> 1 salida (sigmoide)",
        "",
        f"#define ENTRADAS {W1.shape[0]}",
        f"#define OCULTAS {W1.shape[1]}",
        "",
        formatear_array("W1", W1),
        formatear_array("b1", b1),
        formatear_array("W2", W2),
        formatear_array("b2", b2),
        "",
        "float sigmoideC(float z) { return 1.0f / (1.0f + exp(-z)); }",
        "",
        "// Devuelve la probabilidad (0.0 a 1.0) de que haya una botella",
        "float inferirBotella(float entrada[ENTRADAS]) {",
        "  float oculta[OCULTAS];",
        "  for (int j = 0; j < OCULTAS; j++) {",
        "    float suma = b1[j];",
        "    for (int i = 0; i < ENTRADAS; i++) {",
        "      suma += entrada[i] * W1[i * OCULTAS + j];",
        "    }",
        "    oculta[j] = suma > 0 ? suma : 0;  // ReLU",
        "  }",
        "",
        "  float salida = b2[0];",
        "  for (int j = 0; j < OCULTAS; j++) {",
        "    salida += oculta[j] * W2[j];",
        "  }",
        "  return sigmoideC(salida);",
        "}",
    ]
 
    codigo = "\n".join(lineas)
    with open(ruta_salida, "w") as f:
        f.write(codigo + "\n")
    print(f"[Exportacion] Guardado en {ruta_salida}")
 
 
def main():
    X, y_suave, y_duro = construir_dataset_destilacion()
 
    if len(X) == 0:
        print("No hay imagenes en dataset_botellas/. Corre primero capturar_dataset_botellas.py")
        return
 
    # Mezcla: el estudiante aprende principalmente del profesor (soft label),
    # con un poco de la etiqueta real como ancla -- esto ES knowledge distillation.
    y_objetivo = ALPHA_DESTILACION * y_suave + (1 - ALPHA_DESTILACION) * y_duro
 
    print(f"[Entrenamiento] {len(X)} ejemplos, red {X.shape[1]}->{NEURONAS_OCULTAS}->1")
    W1, b1, W2, b2 = entrenar_estudiante(X, y_objetivo)
 
    # Precision simple contra la etiqueta real (umbral 0.5)
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    pred = sigmoide(a1 @ W2 + b2).flatten()
    precision = np.mean((pred > 0.5).astype(float) == y_duro)
    print(f"[Evaluacion] Precision del estudiante vs etiqueta real: {precision:.1%}")
 
    exportar_a_c(W1, b1, W2, b2)
 
 
if __name__ == "__main__":
    main()