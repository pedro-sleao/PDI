from flask import Flask, render_template, Response
from camera import get_stream
from vision import count_vehicles, detect_and_read_plate
import cv2

app = Flask(__name__)

def generate_frames():
    cap = get_stream()
    if cap is None:
        return

    frame_id = 0
    last_plate = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        frame = cv2.resize(frame, (960, 540))

        frame, vehicle_count = count_vehicles(frame)

        frame, plate_text = detect_and_read_plate(frame)
        if plate_text:
            last_plate = plate_text

        # Overlay
        cv2.putText(frame, f"Vehicles: {vehicle_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        if last_plate:
            cv2.putText(frame, f"Plate: {last_plate}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

        # Encode JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG stream
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes +
               b"\r\n")

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
