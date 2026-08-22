"""On-screen performance / debug overlay."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .ar_renderer import _blend_rect
from .config import DisplayMode


def draw_perf_panel(
    frame: np.ndarray,
    fps: float,
    num_targets: int,
    device_label: str,
    mode: DisplayMode,
    pending: int = 0,
    raw_targets: int | None = None,
) -> np.ndarray:
    """
    Translucent status panel in the top-left corner.

    Shows FPS, target count, compute device and current HUD mode.
    """
    mode_name = {
        DisplayMode.DETECTION: "DETECTION",
        DisplayMode.AR_HUD: "AR HUD",
        DisplayMode.CLEAN: "CLEAN",
    }[mode]

    lines = [
        "AUGMENTED VISION",
        f"FPS     {fps:5.1f}",
        f"Targets {num_targets:5d}",
    ]
    if pending > 0:
        lines.append(f"Pending {pending:5d}")
    if raw_targets is not None and raw_targets != num_targets:
        lines.append(f"Raw     {raw_targets:5d}")
    lines.extend(
        [
            f"Device  {device_label}",
            f"Mode    {mode_name}",
            "Keys: H mode | S shot | Q quit",
        ]
    )

    pad_x, pad_y = 12, 12
    line_h = 18
    panel_w = 250
    panel_h = pad_y * 2 + line_h * len(lines) + 4

    _blend_rect(frame, 10, 10, 10 + panel_w, 10 + panel_h, (16, 18, 22), 0.55)
    cv2.rectangle(
        frame,
        (10, 10),
        (10 + panel_w, 10 + panel_h),
        (0, 255, 180),
        1,
        cv2.LINE_AA,
    )

    y = 10 + pad_y + 12
    for i, text in enumerate(lines):
        color: Tuple[int, int, int] = (0, 255, 180) if i == 0 else (220, 230, 235)
        scale = 0.48 if i == 0 else 0.42
        weight = 2 if i == 0 else 1
        cv2.putText(
            frame,
            text,
            (10 + pad_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            weight,
            cv2.LINE_AA,
        )
        y += line_h

    return frame
