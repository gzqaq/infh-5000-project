import functools as ft

import numpy as np


def mask_from_box_coordinates(coordinates: np.ndarray, img: np.ndarray) -> np.ndarray:
    assert coordinates.shape == (4,)

    ul_br_coords = np.round(coordinates).astype(np.int32)
    ul, br = np.split(ul_br_coords, 2)
    width = br[0] - ul[0]
    height = br[1] - ul[1]
    mask = np.zeros_like(img)
    mask[ul[1] :, ul[0] :][:height, :width] = 1

    return mask


def combine_masks(masks: list[np.ndarray]) -> np.ndarray:
    assert len(masks) > 0
    return ft.reduce(np.logical_or, masks, np.zeros_like(masks[0]))
