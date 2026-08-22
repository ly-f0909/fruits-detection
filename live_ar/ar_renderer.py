"""Sci-fi AR HUD drawing primitives — pure OpenCV, no heavy GUI deps."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .config import AppConfig, DisplayMode, color_for_label, get_nutrition_info
from .detector import Detection


Color = Tuple[int, int, int]


def _blend_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Color,
    alpha: float,
) -> None:
    """Fill a rectangle with a translucent solid color via ``addWeighted``."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_corner_reticle(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Color,
    length: int,
    thickness: int,
) -> None:
    """Four targeting brackets (no center crosshair — keeps the view calm)."""
    length = max(8, min(length, (x2 - x1) // 3, (y2 - y1) // 3))

    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def _draw_compact_card(
    frame: np.ndarray,
    x: int,
    y: int,
    det: Detection,
    color: Color,
    config: AppConfig,
) -> None:
    """Compact label: name · conf + calories only."""
    w, h = config.card_width, config.card_height
    fh, fw = frame.shape[:2]
    x = int(np.clip(x, 8, fw - w - 8))
    y = int(np.clip(y, 8, fh - h - 8))

    _blend_rect(frame, x, y, x + w, y + h, (18, 22, 28), config.hud_alpha)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + 3, y + h), color, -1, cv2.LINE_AA)

    title = f"{det.label}  {det.confidence:.0%}"
    nutrition = get_nutrition_info(det.label)

    cv2.putText(
        frame,
        title,
        (x + 12, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 245, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        nutrition,
        (x + 12, y + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 230, 200),
        1,
        cv2.LINE_AA,
    )


def draw_simple_boxes(
    frame: np.ndarray,
    detections: List[Detection],
    config: AppConfig,
) -> np.ndarray:
    """Detection mode: box + label + calories."""
    out = frame
    for det in detections:
        color = color_for_label(det.label)
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        tag = f"{det.label} {det.confidence:.0%}"
        nutrition = get_nutrition_info(det.label)
        ty = max(36, y1 - 10)
        cv2.putText(
            out,
            tag,
            (x1, ty - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            nutrition,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 230, 200),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_ar_hud(
    frame: np.ndarray,
    detections: List[Detection],
    config: AppConfig,
) -> np.ndarray:
    """AR HUD: corner brackets + compact card parked above the box (no leader lines)."""
    out = frame
    for det in detections:
        color = color_for_label(det.label)
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        _draw_corner_reticle(
            out,
            x1,
            y1,
            x2,
            y2,
            color,
            config.reticle_len,
            config.reticle_thickness,
        )

        card_x = x1
        card_y = y1 - config.card_height - 8
        _draw_compact_card(out, card_x, card_y, det, color, config)
    return out


def render_detections(
    frame: np.ndarray,
    detections: List[Detection],
    mode: DisplayMode,
    config: AppConfig,
) -> np.ndarray:
    """Dispatch drawing based on the active display mode."""
    if mode is DisplayMode.DETECTION:
        return draw_simple_boxes(frame, detections, config)
    return draw_ar_hud(frame, detections, config)
