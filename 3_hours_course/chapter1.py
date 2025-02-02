import cv2
import numpy as np

kernel = np.ones((5,5), np.uint8)
img = cv2.imread("../resources/images/lena.png")

# Basic and important functions
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgBlur = cv2.GaussianBlur(img, (7,7), 0) # kernel size (7, 7)
imgCanny = cv2.Canny(img, 100, 100)
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

#cv2.imshow("Lena", img)
#cv2.imshow("Lena Gray", imgGray)
#cv2.imshow("Lena Blur", imgBlur)
cv2.imshow("Lena Canny", imgCanny)
cv2.imshow("Lena Dialation", imgDilation)
cv2.imshow("Lena Eroded", imgEroded)

cv2.waitKey(0)



""" Capturar frames de video
#cap = cv2.VideoCapture("resources/rua.mp4")
cap = cv2.VideoCapture(0) # webcam
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)

while True:
    success, frame = cap.read()
    cv2.imshow( "Rua", frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
"""

