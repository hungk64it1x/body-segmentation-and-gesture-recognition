import cv2
import numpy as np
import matplotlib.pyplot as plt

def refinement(image):
    image = image.copy()
    areas = []
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(image.shape[:2], dtype=image.dtype)

    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
    max_cnt = np.argmax(areas)
    cv2.drawContours(mask, [contours[max_cnt]], 0, (255), -1)
    result = cv2.bitwise_and(image,image, mask= mask)
    return result

