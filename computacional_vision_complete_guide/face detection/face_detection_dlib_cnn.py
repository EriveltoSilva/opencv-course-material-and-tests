# Facial Detection with HOG
import dlib
import cv2

img = cv2.imread("../../resources/images/people2.jpg")

detector_face_cnn = dlib.cnn_face_detection_model_v1()
detections = detector_face_cnn(img, 4)

print(detections, len(detections))
for face in detections:
    l, t, r, b, confidence = face.rect.left(), face.rect.top(), face.rect.right() , face.rect.bottom(), face.confidence
    cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)


cv2.imshow("Original", img)
cv2.waitKey(0)