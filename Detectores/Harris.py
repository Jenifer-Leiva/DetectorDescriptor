import numpy as np
import cv2 as cv
filename = 'chessboard1.jpg'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

ventana_w = 800
ventana_h = 600
h, w = img.shape[:2]
escala = min(ventana_w / w, ventana_h / h)
nuevo_w = int(w * escala)
nuevo_h = int(h * escala)
img_redimensionada = cv.resize(img, (nuevo_w, nuevo_h))
cv.namedWindow("Imagen", cv.WINDOW_NORMAL)
cv.resizeWindow("Imagen", ventana_w, ventana_h)
cv.imshow("Imagen", img_redimensionada)
cv.waitKey(0)
cv.destroyAllWindows()


if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()