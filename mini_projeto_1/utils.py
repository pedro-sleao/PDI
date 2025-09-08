import cv2
import numpy as np

def get_labels(img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    indices_hue = np.uint8(179*labels/np.max(labels))
    canal_vazio = 255*np.ones_like(indices_hue)
    img_indices = cv2.merge([indices_hue, canal_vazio, canal_vazio])

    img_final = cv2.cvtColor(img_indices, cv2.COLOR_HSV2RGB)
    # set bg label to black
    img_final[indices_hue==0] = 0

    return labels
