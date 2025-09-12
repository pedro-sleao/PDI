import math
import numpy as np
import cv2

def draw_color_preview(img, color):
    cv2.rectangle(img, (10, 10), (30, 30), color, -1) 

def draw_reference_points(img, points):
    """Desenha pontos e índices na tela"""
    for i, pt in enumerate(points):
        cv2.circle(img, tuple(map(int, pt)), 5, (0,255,0), -1)
        cv2.putText(img, str(i+1), (int(pt[0])+5, int(pt[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
def get_labels(img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    indices_hue = np.uint8(179*labels/np.max(labels))
    canal_vazio = 255*np.ones_like(indices_hue)
    img_indices = cv2.merge([indices_hue, canal_vazio, canal_vazio])

    img_final = cv2.cvtColor(img_indices, cv2.COLOR_HSV2RGB)
    # set bg label to black
    img_final[indices_hue==0] = 0

    return labels

# Callback functions
def mouse_calibrating_cb(event,x,y,flags,param):
    global calibrating, referencePoints

    if event == cv2.EVENT_LBUTTONDOWN and calibrating:
        referencePoints.append([x, y])

        if len(referencePoints) == 4:
            calibrating = False

def mouse_painting_cb(event,x,y,flags,param):
    global output_img, painting, paintColor

    if event == cv2.EVENT_LBUTTONDOWN and painting:
        target_label = labels[y, x]
        mask = (labels == target_label)
        if target_label != 0:
            output_img[mask] = paintColor

def on_change(val):
    global paintColor, colorList
    paintColor = colorList[val]


inputImage1 = cv2.imread("images/BG.jpg")
inputImage1 = cv2.cvtColor(inputImage1, cv2.COLOR_BGR2RGB)

rows1, cols1 = inputImage1.shape[:2]
pts1 = np.float32(
    [[0,0],
    [cols1,0],
    [cols1, rows1],
    [0, rows1]]
)

referencePoints = []

output_img = inputImage1.copy()
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
output_img = cv2.threshold(output_img, 65, 255, cv2.THRESH_BINARY)[1]

labels = get_labels(output_img)

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
cv2.setMouseCallback("coloredImage", mouse_painting_cb)

cv2.namedWindow("warpedImage", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("warpedImage", mouse_calibrating_cb)

cv2.createTrackbar("Pick color", "coloredImage", 0, len(colorList)-1, on_change)

paintColor = colorList[0]
painting = False
fullScreen = False
calibrating = False

image = np.zeros((rows1, cols1, 3), dtype=np.uint8)

# --- Loop principal ---
while True:
    image[:] = (0, 0, 0)

    if len(referencePoints) == 4:
        M = cv2.getPerspectiveTransform(pts1, np.float32(referencePoints))
        image = cv2.warpPerspective(output_img, M, (cols1, rows1), borderMode=cv2.BORDER_TRANSPARENT)
    else:
        image = output_img.copy()

    # Mostra pontos de calibração
    if calibrating or len(referencePoints) == 4:
        draw_reference_points(image, referencePoints)
    if calibrating:
        cv2.putText(image, "Click 4 points | 1 - SE, 2 - SD, 3 - ID, 4 - IE", (40,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("warpedImage", image)
    cv2.imshow("coloredImage", output_img)
    
    draw_color_preview(output_img, paintColor)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        calibrating = not calibrating
        referencePoints = []

    if key == ord("p"):
        painting = not painting

    if key == ord("f"):
        if not fullScreen:
            cv2.setWindowProperty("warpedImage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("warpedImage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        fullScreen = not fullScreen

    if key == ord("q"):
        break

cv2.destroyAllWindows()
