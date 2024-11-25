from pathlib import Path

import cv2
import numpy as np

XMAS_DIR = Path(__file__).parent.resolve()
CFG_DIR = XMAS_DIR / "configs"
HAT_DIR = XMAS_DIR.parent.parent / "assets" / "xmas_hats"


def read_hats() -> list[np.ndarray]:
    res = []
    for hat_path in HAT_DIR.iterdir():
        img = cv2.cvtColor(cv2.imread(hat_path, -1), cv2.COLOR_BGRA2RGBA)
        img[..., :3] = np.where(
            np.logical_and(img[..., -1:] > 200, img[..., :3] == 0), 1, img[..., :3]
        )
        img[..., 3] = np.where(img[..., 3] > 127, 255, 0)
        res.append(img)

    return res


def read_configs() -> list[Path]:
    return list(CFG_DIR.iterdir())
