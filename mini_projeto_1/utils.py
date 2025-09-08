import cv2
import numpy as np

def get_labels(img):
    img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_b = cv2.threshold(img_g, 50, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_b)
    indices_hue = np.uint8(179*labels/np.max(labels))
    canal_vazio = 255*np.ones_like(indices_hue)
    img_indices = cv2.merge([indices_hue, canal_vazio, canal_vazio])

    img_final = cv2.cvtColor(img_indices, cv2.COLOR_HSV2RGB)
    # set bg label to black
    img_final[indices_hue==0] = 0

    return labels
