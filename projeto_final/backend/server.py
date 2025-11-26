from flask import Flask, render_template, Response
from camera import get_snapshot
from vision import count_vehicles
from vision import detect_and_read_plate
import cv2

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/count-vehicles")
def count_vehicles_route():
    frame = get_snapshot()
    if frame is None:
        return "Erro ao obter imagem", 500

    processed, count = count_vehicles(frame)

    ret, jpeg = cv2.imencode(".jpg", processed)
    if not ret:
        return "Erro ao codificar imagem", 500

    response = Response(jpeg.tobytes(), mimetype="image/jpeg")
    response.headers["X-Vehicle-Count"] = str(count)
    return response

@app.route("/read-plate")
def read_plate_route():
    frame = get_snapshot()
    if frame is None:
        return "Erro ao obter imagem", 500

    processed, plate_text = detect_and_read_plate(frame)

    ret, jpeg = cv2.imencode(".jpg", processed)
    if not ret:
        return "Erro ao codificar imagem", 500

    response = Response(jpeg.tobytes(), mimetype="image/jpeg")
    response.headers["X-Plate-Text"] = plate_text
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
