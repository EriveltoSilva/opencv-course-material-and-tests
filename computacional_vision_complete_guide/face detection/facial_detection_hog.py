# Facial Detection with HOG
import dlib
import cv2

img = cv2.imread("../../resources/images/people2.jpg")

detector_face_hog = dlib.get_frontal_face_detector()
detections = detector_face_hog(img, 4)

print(detections, len(detections))
for face in detections:
    l, t, r, b = face.left(), face.top(), face.right() , face.bottom()
    cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)


cv2.imshow("Original", img)
cv2.waitKey(0)