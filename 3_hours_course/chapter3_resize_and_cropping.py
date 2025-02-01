""" Resizing and cropping - Opencv Convention"""

import cv2
import numpy as np


img = cv2.imread("resources/lambo.png")

# Resize
imgResized = cv2.resize(img, (300, 200))
print(imgResized.shape)

imgCropped = img[0:200, 200:500]
print(imgCropped.shape)


# apresentation
cv2.imshow("----- Original ------",img)
cv2.imshow("----- Resized ------",imgResized)
cv2.imshow("----- Cropped ------",imgCropped)

cv2.waitKey(0)
