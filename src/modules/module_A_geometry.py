"""Clinical morphology extraction from segmentation masks."""

from __future__ import annotations

import cv2
import numpy as np


def _height_from_mask(binary_mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    _, _, _, h = cv2.boundingRect(cnt)
    return int(h)


def extract_morphology_metrics(mask_np: np.ndarray) -> np.ndarray:
    vals = np.unique(mask_np)
    if len(vals) < 2:
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)

    non_zero = [v for v in vals if v != 0]
    disc_val = max(non_zero)
    cup_val = min(non_zero)

    cup_mask = (mask_np == cup_val).astype(np.uint8)
    disc_mask = (mask_np == disc_val).astype(np.uint8)

    cup_h = _height_from_mask(cup_mask)
    disc_h = _height_from_mask(disc_mask)
    vcdr = cup_h / disc_h if disc_h > 0 else 0.5

    cup_area = float(cup_mask.sum())
    disc_area = float(disc_mask.sum())
    area_ratio = cup_area / disc_area if disc_area > 0 else 0.0

    return np.array([vcdr, area_ratio, disc_area], dtype=np.float32)
