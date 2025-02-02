""" Capturar frames de video"""

import cv2

cap = cv2.VideoCapture(0) # webcam
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)

face_detector = cv2.CascadeClassifier('../../resources/haarcascades/haarcascade_frontalface_default.xml')

while True:
    success, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(gray_image)
    for (x,y,w,h) in detections:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
