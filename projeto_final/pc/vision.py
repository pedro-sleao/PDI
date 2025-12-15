import cv2
from ultralytics import YOLO
import easyocr

vehicles_model = YOLO("models/yolo11n.pt")
plate_model = YOLO("models/best.pt")

reader = easyocr.Reader(['pt'])

def count_vehicles(frame):
    results = vehicles_model(frame, verbose=False, classes=[1,2,3,4,5,6,7])[0]
    vehicle_count = len(results.boxes)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, vehicle_count

def detect_and_read_plate(frame):
    results = plate_model(frame, verbose=False)[0]

    if len(results.boxes) == 0:
        return frame, ""

    best_box = max(results.boxes, key=lambda b: float(b.conf[0]))

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    conf = float(best_box.conf[0])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    crop = frame[y1:y2, x1:x2]

    try:
        result = reader.readtext(crop, detail=0)
        plate_text = result[0] if len(result) > 0 else ""
    except:
        plate_text = ""

    return frame, plate_text

def track_vehicles(frame):
    results = vehicles_model.track(
        frame,
        persist=True,
        classes=[1,2,3,4,5,6,7],
        verbose=False
    )[0]

    vehicle_ids = set()

    for box in results.boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        vehicle_ids.add(track_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} ID:{track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame, vehicle_ids
