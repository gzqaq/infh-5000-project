import cv2
import numpy as np


def wear_hat(img: np.ndarray, face: np.ndarray, hat: np.ndarray) -> np.ndarray:
    scale = face[3] / hat.shape[0] * 2
    hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale)
    x_offset = int(face[0] + face[2] / 2 - hat.shape[1] / 2)
    y_offset = int(face[1] - hat.shape[0] / 2)
    x1 = max(x_offset, 0)
    x2 = min(x_offset + hat.shape[1], img.shape[1])
    y1 = max(y_offset, 0)
    y2 = min(y_offset + hat.shape[0], img.shape[0])
    hat_x1 = max(0, -x_offset)
    hat_x2 = hat_x1 + x2 - x1
    hat_y1 = max(0, -y_offset)
    hat_y2 = hat_y1 + y2 - y1

    alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
    alpha = 1 - alpha_h

    for channel in range(3):
        img[y1:y2, x1:x2, channel] = (
            alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, channel]
            + alpha * img[y1:y2, x1:x2, channel]
        )

    return img
