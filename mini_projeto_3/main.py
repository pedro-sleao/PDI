import cv2
import numpy as np

def detect_and_remove_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    mask = np.zeros_like(gray)

    for (x, y, w, h) in faces:
        # Preenche a região do rosto na máscara (em branco)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    kernel = np.ones((60, 60), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)

    inpainted = cv2.inpaint(frame, mask_dilated, 3, cv2.INPAINT_NS)

    return inpainted

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

while True:
    result, video_frame = video_capture.read()
    if not result:
        break
    
    frame_no_face = detect_and_remove_face(video_frame)

    cv2.imshow("Removed Faces (Press 'q' to Quit)", frame_no_face)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
