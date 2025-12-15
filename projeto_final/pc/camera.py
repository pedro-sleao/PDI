import requests
import cv2
import numpy as np

ESP32_CAM_URL = "http://192.168.0.101/capture"

def get_snapshot():
    try:
        response = requests.get(ESP32_CAM_URL, timeout=5)

        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        print("Erro ao acessar a câmera:", response.status_code)
        return None

    except Exception as e:
        print("Erro:", e)
        return None
