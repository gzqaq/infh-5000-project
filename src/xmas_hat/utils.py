from pathlib import Path

import cv2
import numpy as np

XMAS_DIR = Path(__file__).parent.resolve()
HAT_DIR = XMAS_DIR.parent.parent / "assets" / "xmas_hats"


def read_hats() -> list[np.ndarray]:
    return [cv2.imread(hat_path, -1) for hat_path in HAT_DIR.iterdir()]
