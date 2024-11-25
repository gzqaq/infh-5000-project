from pathlib import Path

import cv2
import numpy as np


def detect_face(img: np.ndarray, config: Path) -> np.ndarray:
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(grey_img)
    face_cascade = cv2.CascadeClassifier(config)
    faces = face_cascade.detectMultiScale(hist)

    return faces
