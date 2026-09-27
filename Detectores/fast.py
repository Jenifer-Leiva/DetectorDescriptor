import numpy as np
import cv2 as cv
from pathlib import Path


ruta = Path(__file__).resolve().parent / "chessboard.jpg"
img = cv.imread(str(ruta))

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

fast=cv.FastFeatureDetector_create(
    threshold=2,
    nonmaxSuppression=True,
    type=cv.FAST_FEATURE_DETECTOR_TYPE_7_12
)

## encontrar dibujar  keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

## imprimir parametros
print("Parámetros de Fast")
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )



# Obtener coordenadas
puntos = np.array(
    [k.pt for k in kp],
    dtype=np.float32
)


# Agrupar puntos cercanos

esquinas = []
distancia_minima = 15
for punto in puntos:
    x, y = punto
    muy_cerca = False
    for esquina in esquinas:
        ex, ey = esquina
        distancia = np.sqrt(
            (x - ex)**2 +
            (y - ey)**2
        )
        if distancia < distancia_minima:
            muy_cerca = True
            break
    if not muy_cerca:
        esquinas.append((x, y))

# Resultado
print("Esquinas estimadas:", len(esquinas))

# Dibujar
for i, (x, y) in enumerate(esquinas):

    x = int(x)
    y = int(y)

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
escala = min(ventana_w / w,ventana_h / h)
nuevo_w = int(w * escala)
nuevo_h = int(h * escala)
img_redimensionada = cv.resize(img, (nuevo_w, nuevo_h))
cv.namedWindow("FAST - Esquinas", cv.WINDOW_NORMAL)
cv.resizeWindow("FAST - Esquinas",ventana_w,ventana_h)
cv.imshow("FAST - Esquinas", img_redimensionada)
cv.waitKey(0)
cv.destroyAllWindows()