import random

import numpy as np

from .face_detection import detect_face
from .utils import read_configs, read_hats
from .wear_hat import wear_hat


def wear_hats(img: np.ndarray, labels: str = "") -> np.ndarray:
    hats = read_hats()

    if "cat" in labels:
        configs = read_configs(["real", "anime"])
    else:
        configs = read_configs()

    for config in configs:
        for face in detect_face(img, config):
            hat = random.choice(hats)
            img = wear_hat(img, face, hat)

    return img
