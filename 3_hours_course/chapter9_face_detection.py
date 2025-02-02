""" Face detection"""
import cv2

face_cascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_frontalface_default.xml")
img = cv2.imread("../resources/images/Erivelto_Perfil-removebg.png")
#img = cv2.resize(img, (700,600))
imgGray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecting faces and drawing the rectangles on it
faces = face_cascade.detectMultiScale(imgGray, 1.2, minNeighbors=5)
print(faces)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)


print(f'Número de faces detectadas:{len(faces)}')

cv2.imshow("Original", img)
cv2.waitKey(0)
