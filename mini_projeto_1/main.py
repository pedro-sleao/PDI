import math
import numpy as np
import cv2
from utils import get_labels

def draw_color_preview(img, color):
    cv2.rectangle(img, (10, 10), (20, 20), color, -1) 

# Callback functions
def draw_circle(event,x,y,flags,param):
    global output_img
    if event == cv2.EVENT_LBUTTONDOWN and painting:
        target_label = labels[y, x]
        mask = (labels == target_label)
        if target_label != 0:
            output_img[mask] = paintColor

def on_change(val):
    global paintColor, colorList

    paintColor = colorList[val]
    
inputImage1 = cv2.imread("images/dinossauro.jpg")
inputImage1 = cv2.cvtColor(inputImage1, cv2.COLOR_BGR2RGB)

labels = get_labels(inputImage1)

output_img = inputImage1.copy()
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
output_img = np.array([[output_img[i][j] if output_img[i][j] < 35 else 255 for j in range(len(output_img[i]))] for i in range(len(output_img))], dtype=np.uint8)
output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)

colorList = [
    (0, 255, 0),     # verde
    (0, 0, 255),     # vermelho
    (255, 0, 0),     # azul
    (255, 255, 0),   # amarelo
    (255, 0, 255),   # magenta
    (0, 255, 255),   # ciano
    (128, 128, 128), # cinza
    (255, 255, 255)  # branco
]

cv2.namedWindow("coloredImage", cv2.WINDOW_NORMAL)
cv2.setMouseCallback('coloredImage', draw_circle)
cv2.createTrackbar("Pick color", "coloredImage", 0, len(colorList)-1, on_change)

paintColor = colorList[0]
painting = False
fullScreen = False

while True:
    cv2.imshow("coloredImage", output_img)
    draw_color_preview(output_img, paintColor)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):
        painting = not painting

    if key == ord("f"):
        if not fullScreen:
            cv2.setWindowProperty("coloredImage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("coloredImage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        fullScreen = not fullScreen

    if key == ord("q"):
        break

cv2.destroyAllWindows()