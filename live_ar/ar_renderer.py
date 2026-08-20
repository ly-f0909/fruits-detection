"""Sci-fi AR HUD drawing primitives — pure OpenCV, no heavy GUI deps."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .config import AppConfig, DisplayMode, color_for_label
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
    """Four targeting brackets instead of a full bounding rectangle."""
    length = max(8, min(length, (x2 - x1) // 3, (y2 - y1) // 3))

    # TL
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    # TR
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    # BL
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    # BR
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)

    # Center crosshair (subtle)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    arm = max(4, length // 3)
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)


def _draw_leader_line(
    frame: np.ndarray,
    anchor: Tuple[int, int],
    card_anchor: Tuple[int, int],
    color: Color,
) -> None:
    """Polyline from object top-center → elbow → HUD card."""
    ax, ay = anchor
    cx, cy = card_anchor
    # Elbow sits above the object, offset sideways toward the card.
    mid_x = ax + int(0.55 * (cx - ax))
    mid_y = min(ay, cy) - 28
    pts = np.array([[ax, ay], [mid_x, mid_y], [cx, cy]], dtype=np.int32)
    cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 3, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)


def _estimate_geometry(det: Detection) -> Tuple[str, str]:
    """
    Lightweight geometric cues from the 2D box (demo-friendly, not metric scale).

    Returns human-readable diameter (px) and aspect ratio strings.
    """
    w, h = det.width, det.height
    diameter_px = 0.5 * (w + h)
    ratio = (w / h) if h > 1e-3 else 0.0
    return f"{diameter_px:.0f}px", f"{ratio:.2f}"


def _draw_hud_card(
    frame: np.ndarray,
    origin: Tuple[int, int],
    det: Detection,
    color: Color,
    config: AppConfig,
) -> None:
    """Semi-transparent info panel at the end of the leader line."""
    x, y = origin
    w, h = config.card_width, config.card_height
    fh, fw = frame.shape[:2]

    # Keep card on-screen
    x = int(np.clip(x, 8, fw - w - 8))
    y = int(np.clip(y, 8, fh - h - 8))

    # Panel background + neon border
    _blend_rect(frame, x, y, x + w, y + h, (18, 22, 28), config.hud_alpha)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    # Top accent bar
    cv2.rectangle(frame, (x, y), (x + w, y + 3), color, -1, cv2.LINE_AA)

    diameter, ratio = _estimate_geometry(det)
    title = det.label.upper()
    conf_pct = int(round(det.confidence * 100))

    cv2.putText(
        frame,
        title,
        (x + 10, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 245, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ID {det.track_id}  ·  {conf_pct}%",
        (x + 10, y + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 190, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Diam {diameter}   Ratio {ratio}",
        (x + 10, y + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 175, 190),
        1,
        cv2.LINE_AA,
    )

    # Confidence bar
    bar_x1, bar_y1 = x + 10, y + h - 16
    bar_x2, bar_y2 = x + w - 10, y + h - 8
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 48, 56), -1)
    fill = int((bar_x2 - bar_x1) * float(np.clip(det.confidence, 0.0, 1.0)))
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill, bar_y2), color, -1)


def _card_origin_for(
    det: Detection,
    frame_shape: Sequence[int],
    card_w: int,
    card_h: int,
    index: int,
) -> Tuple[int, int]:
    """Place HUD cards above objects with a slight fan-out to reduce overlap."""
    fh, fw = frame_shape[:2]
    ax, ay = det.top_center
    # Alternate left/right offsets by track order
    side = 1 if (index % 2 == 0) else -1
    ox = int(ax + side * (70 + 15 * (index % 3)))
    oy = int(ay - card_h - 48)
    ox = int(np.clip(ox, 8, fw - card_w - 8))
    oy = int(np.clip(oy, 8, fh - card_h - 8))
    return ox, oy


def draw_simple_boxes(
    frame: np.ndarray,
    detections: List[Detection],
    config: AppConfig,
) -> np.ndarray:
    """Classic detection mode: rectangle + label."""
    out = frame
    for det in detections:
        color = color_for_label(det.label)
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        tag = f"{det.label} {det.confidence:.0%}"
        cv2.putText(
            out,
            tag,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def draw_ar_hud(
    frame: np.ndarray,
    detections: List[Detection],
    config: AppConfig,
) -> np.ndarray:
    """Full AR HUD: reticle, leader line, translucent info card."""
    out = frame
    for i, det in enumerate(detections):
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

        card_xy = _card_origin_for(
            det, out.shape, config.card_width, config.card_height, i
        )
        anchor = (int(det.cx), int(det.y1))
        # Leader attaches to the bottom-center of the card
        line_end = (card_xy[0] + config.card_width // 2, card_xy[1] + config.card_height)
        # Prefer attaching to the nearer vertical edge of the card
        if abs(anchor[0] - card_xy[0]) < abs(anchor[0] - (card_xy[0] + config.card_width)):
            line_end = (card_xy[0], card_xy[1] + 20)
        else:
            line_end = (card_xy[0] + config.card_width, card_xy[1] + 20)

        _draw_leader_line(out, anchor, line_end, color)
        _draw_hud_card(out, card_xy, det, color, config)
    return out


def render_detections(
    frame: np.ndarray,
    detections: List[Detection],
    mode: DisplayMode,
    config: AppConfig,
) -> np.ndarray:
    """Dispatch drawing based on the active display mode."""
    if mode is DisplayMode.CLEAN:
        return frame
    if mode is DisplayMode.DETECTION:
        return draw_simple_boxes(frame, detections, config)
    return draw_ar_hud(frame, detections, config)
