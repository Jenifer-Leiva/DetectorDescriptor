import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('chessboard1.jpg', cv.IMREAD_GRAYSCALE) # `<opencv_root>/samples/data/blox.jpg`

## Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

## find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

## Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

cv.imwrite('fast_true.png', img2)

## Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

cv.imwrite('fast_false.png', img3)



ventana_w = 800
ventana_h = 600
h, w = img.shape[:2]
escala = min(ventana_w / w, ventana_h / h)
nuevo_w = int(w * escala)
nuevo_h = int(h * escala)
img_redimensionada = cv.resize(img3, (nuevo_w, nuevo_h))
cv.namedWindow("Imagen", cv.WINDOW_NORMAL)
cv.resizeWindow("Imagen", ventana_w, ventana_h)
cv.imshow("Imagen", img_redimensionada)
cv.waitKey(0)
cv.destroyAllWindows()


if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()