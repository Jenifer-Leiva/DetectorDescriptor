import numpy as np
import cv2 as cv
from pathlib import Path

ruta = Path(__file__).resolve().parent / "chessboard.jpg"
img = cv.imread(str(ruta))

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Param
blockSize = 2
ksize = 3
k = 0.04

harris = cv.cornerHarris(
    gray,
    blockSize,
    ksize,
    k
)

#dilatacion bordes
harris = cv.dilate(harris,None)
# umbral valor optimo
umbral = 0.01 * harris.max()


print("Parámetros de Harris")
print("Block Size:", blockSize)
print("Kernel Size (ksize):", ksize)
print("Harris k:", k)
print("Threshold:", 0.01)
print("Threshold absoluto:", umbral)

# Obtener coordenadas de los puntos detectados
ys, xs = np.where(harris > umbral)
puntos = list(zip(xs, ys))

# Agrupar puntos cercanos
esquinas = []
distancia_minima = 15
for x, y in puntos:
    muy_cerca = False
    for ex, ey in esquinas:
        distancia = np.sqrt(
            (x - ex) ** 2 +
            (y - ey) ** 2
        )
        if distancia < distancia_minima:
            muy_cerca = True
            break
    if not muy_cerca:
        esquinas.append((x, y))

print("Puntos detectados por Harris:", len(puntos))
print("Esquinas estimadas:", len(esquinas))

# Dibujar
for i, (x, y) in enumerate(esquinas):
    cv.circle(
        img,
        (x, y),
        5,
        (0, 0, 255),
        -1
    )
    cv.putText(
        img,
        str(i + 1),
        (x + 5, y - 5),
        cv.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )

# Mostrar
ventana_w = 800
ventana_h = 600
h, w = img.shape[:2]
escala = min(ventana_w / w,ventana_h)
nuevo_w = int(w * escala)
nuevo_h = int(h * escala)
img_redimensionada = cv.resize(img,(nuevo_w, nuevo_h))
cv.namedWindow("Harris - Esquinas", cv.WINDOW_NORMAL)
cv.resizeWindow("Harris - Esquinas",ventana_w,ventana_h)
cv.imshow("Harris - Esquinas",img_redimensionada)
cv.waitKey(0)
cv.destroyAllWindows()