"""Lightweight 2D geometry helpers for verification filters."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def polygon_box_overlap_ratio(polygon: np.ndarray, box: Box, frame_shape: Tuple[int, int]) -> float:
    """
    Fraction of ``box`` area covered by ``polygon`` (convex hull of hand landmarks).

    Uses binary masks — fast enough at 720p for real-time use.
    """
    h, w = frame_shape[:2]
    fruit_area = box_area(box)
    if fruit_area < 1.0:
        return 0.0

    mask_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_poly, polygon.astype(np.int32), 255)

    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    mask_box = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask_box, (x1, y1), (x2, y2), 255, thickness=-1)

    overlap = cv2.bitwise_and(mask_poly, mask_box)
    return float(np.count_nonzero(overlap)) / fruit_area


def landmarks_to_hull(
    landmarks: Sequence,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    pts = np.array(
        [[int(lm.x * frame_width), int(lm.y * frame_height)] for lm in landmarks],
        dtype=np.int32,
    )
    if len(pts) < 3:
        return pts.reshape(-1, 1, 2)
    return cv2.convexHull(pts)
