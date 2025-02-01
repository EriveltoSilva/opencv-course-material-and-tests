""" Shapes Resizing and Texts"""

import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)

cv2.line(img, (0,0), (img.shape[1], img.shape[0]), (0,255,0), thickness=3)
cv2.rectangle(img, (0,0), (250, 350), (0,0,255), thickness=2)
cv2.circle(img, (400, 50), 70, (255, 255, 0), thickness=4)
cv2.putText(img, "Hello World", (300, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,150,0), thickness=2)

#img[50:100, 100:300] = 255, 0, 0
#print(img)


cv2.imshow("Image", img)

cv2.waitKey(0)
