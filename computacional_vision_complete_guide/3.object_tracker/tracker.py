"""object tracker"""
import cv2

#tracker = cv2.TrackerKCF_create() # Ideal para videos e cenários mais lentos
tracker = cv2.TrackerCSRT_create() # Ideal para videos rápidos


#video = cv2.VideoCapture("../../resources/videos/race.mp4")
video = cv2.VideoCapture("../../resources/videos/street.mp4")
ok, frame = video.read()
bbox = cv2.selectROI(frame)

ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break
    ok, bbox = tracker.update(frame)

    if ok:
        (x, y, w,h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
    else:
        cv2.putText(frame, "Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
