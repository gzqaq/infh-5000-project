import random

import numpy as np

from .face_detection import detect_face
from .utils import read_configs, read_hats
from .wear_hat import wear_hat


def wear_hats(img: np.ndarray) -> np.ndarray:
    hats = read_hats()
    for config in read_configs():
        for face in detect_face(img, config):
            hat = random.choice(hats)
            img = wear_hat(img, face, hat)

    return img
