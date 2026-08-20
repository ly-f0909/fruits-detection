"""Webcam capture helpers with graceful failure messaging."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class WebcamCapture:
    """Thin wrapper around ``cv2.VideoCapture`` with sizing + health checks."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the device; try AVFoundation on macOS when the default backend fails."""
        backends = [cv2.CAP_ANY]
        # Prefer AVFoundation on Apple Silicon / macOS for more reliable device open.
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.insert(0, cv2.CAP_AVFOUNDATION)

        last_err = ""
        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency
                ok, _ = cap.read()
                if ok:
                    self._cap = cap
                    return
                cap.release()
                last_err = "opened but failed to read a frame"
            else:
                last_err = "isOpened() returned False"
                cap.release()

        raise CameraError(
            f"Cannot open camera index {self.index} ({last_err}).\n"
            "Tips:\n"
            "  • Grant camera permission to Terminal / Python / Cursor\n"
            "  • Try another index:  python main.py --camera 1\n"
            "  • Close apps that are already using the webcam"
        )

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
