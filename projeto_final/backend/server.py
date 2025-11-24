import requests
import cv2
from ultralytics import YOLO
import numpy as np
from flask import Flask, render_template, Response

ESP32_CAM_URL = "http://192.168.1.3/capture"

app = Flask(__name__)

model = YOLO("models/yolo11n.pt")

def get_snapshot():
    try:
        response = requests.get(ESP32_CAM_URL, timeout=5)

        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        else:
            print("Erro ao acessar a câmera:", response.status_code)
            return None

    except Exception as e:
        print("Erro:", e)
        return None

def process_frame(frame):
    results = model(frame, verbose=False, classes=[1,2,3,5,6,7])[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        # desenhar retângulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # escrever texto acima
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    return frame


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/snapshot")
def snapshot():
    frame = get_snapshot()

    if frame is None:
        return "Erro ao obter imagem", 500

    processed = process_frame(frame)

    ret, jpeg = cv2.imencode(".jpg", processed)

    if not ret:
        return "Erro ao codificar imagem", 500

    return Response(jpeg.tobytes(), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
