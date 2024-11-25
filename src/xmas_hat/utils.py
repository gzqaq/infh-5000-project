from pathlib import Path

import cv2
import numpy as np

XMAS_DIR = Path(__file__).parent.resolve()
CFG_DIR = XMAS_DIR / "configs"
HAT_DIR = XMAS_DIR.parent.parent / "assets" / "xmas_hats"


def read_hats() -> list[np.ndarray]:
    return [
        cv2.cvtColor(cv2.imread(hat_path, -1), cv2.COLOR_BGRA2RGBA)
        for hat_path in HAT_DIR.iterdir()
    ]


def read_configs() -> list[Path]:
    return list(CFG_DIR.iterdir())
