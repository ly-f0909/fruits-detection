"""MediaPipe Hands — reject produce boxes dominated by hand regions."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Protocol

import cv2
import numpy as np

from .config import AppConfig
from .detector import Detection
from .geometry import landmarks_to_hull, polygon_box_overlap_ratio

WARN_MSG = "[WARN] MediaPipe hands unavailable, skipping hand filter"

ROOT = Path(__file__).resolve().parents[1]
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


class _HandBackend(Protocol):
    """Minimal interface shared by Solutions- and Tasks-based backends."""

    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        ...

    def close(self) -> None:
        ...


class _SolutionsHandsBackend:
    """Legacy Solutions API (mediapipe.python.solutions or mp.solutions)."""

    def __init__(self, hands: Any) -> None:
        self._hands = hands

    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        return [
            landmarks_to_hull(hand_lms.landmark, w, h)
            for hand_lms in result.multi_hand_landmarks
        ]

    def close(self) -> None:
        self._hands.close()


class _TasksHandLandmarkerBackend:
    """MediaPipe Tasks API (mediapipe >= 0.10.31 / 1.x)."""

    def __init__(self, landmarker: Any) -> None:
        self._landmarker = landmarker
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms = int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return []
        return [
            landmarks_to_hull(hand_lms, w, h) for hand_lms in result.hand_landmarks
        ]

    def close(self) -> None:
        self._landmarker.close()


def _ensure_hand_model() -> Path:
    HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HAND_MODEL_PATH.exists() and HAND_MODEL_PATH.stat().st_size > 100_000:
        return HAND_MODEL_PATH
    try:
        print(f"[INFO] Downloading hand landmarker model → {HAND_MODEL_PATH}")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    except Exception as exc:
        raise OSError(
            f"Hand model missing at {HAND_MODEL_PATH}. Download manually:\n"
            f"  curl -L '{HAND_MODEL_URL}' -o models/hand_landmarker.task"
        ) from exc
    return HAND_MODEL_PATH


def _load_solutions_backend(config: AppConfig) -> _HandBackend:
    """
    Preferred path for mediapipe 0.10.14–0.10.30:
      mediapipe.python.solutions.hands
    """
    import mediapipe as mp  # noqa: F401
    from mediapipe.python.solutions import drawing_utils as mp_drawing  # noqa: F401
    from mediapipe.python.solutions import hands as mp_hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )
    return _SolutionsHandsBackend(hands)


def _load_legacy_solutions_backend(config: AppConfig) -> _HandBackend:
    """Fallback: mp.solutions.hands (older wheels)."""
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )
    return _SolutionsHandsBackend(hands)


def _load_tasks_backend(config: AppConfig) -> _HandBackend:
    """Fallback: MediaPipe Tasks HandLandmarker (mediapipe >= 0.10.31 / 1.x)."""
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    model_path = _ensure_hand_model()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = HandLandmarker.create_from_options(options)
    return _TasksHandLandmarkerBackend(landmarker)


def _create_hand_backend(config: AppConfig) -> Optional[_HandBackend]:
    loaders = (
        ("mediapipe.python.solutions", _load_solutions_backend),
        ("mp.solutions.hands", _load_legacy_solutions_backend),
        ("mediapipe.tasks", _load_tasks_backend),
    )
    for label, loader in loaders:
        try:
            backend = loader(config)
            print(f"[INFO] Hand occlusion filter enabled ({label})")
            return backend
        except (ImportError, ModuleNotFoundError, AttributeError, OSError, RuntimeError, ValueError):
            continue
        except Exception:
            continue
    return None


class HandOcclusionFilter:
    """
    Lightweight hand detector for false-positive suppression.

    Tries, in order:
      1. mediapipe.python.solutions.hands  (0.10.14+ legacy path)
      2. mp.solutions.hands                (older installs)
      3. mediapipe.tasks HandLandmarker    (0.10.31+ / 1.x)

    On total failure: prints a warning and disables itself without crashing.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = config.enable_hand_filter
        self._backend: Optional[_HandBackend] = None

        if not self.enabled:
            return

        try:
            self._backend = _create_hand_backend(config)
            if self._backend is None:
                print(WARN_MSG)
                self.enabled = False
        except (ImportError, ModuleNotFoundError, AttributeError, OSError, RuntimeError, ValueError):
            print(WARN_MSG)
            self._backend = None
            self.enabled = False
        except Exception:
            print(WARN_MSG)
            self._backend = None
            self.enabled = False

    def close(self) -> None:
        if self._backend is None:
            return
        try:
            self._backend.close()
        except Exception:
            pass
        finally:
            self._backend = None

    def _hand_hulls(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        if self._backend is None:
            return []
        try:
            return self._backend.process(frame_bgr)
        except Exception:
            return []

    def filter(self, frame_bgr: np.ndarray, detections: List[Detection]) -> List[Detection]:
        if not self.enabled or self._backend is None or not detections:
            return detections

        try:
            hulls = self._hand_hulls(frame_bgr)
            if not hulls:
                return detections

            kept: List[Detection] = []
            reject_ratio = self.config.hand_overlap_reject
            shape = frame_bgr.shape

            for det in detections:
                box = (det.x1, det.y1, det.x2, det.y2)
                max_overlap = 0.0
                for hull in hulls:
                    ratio = polygon_box_overlap_ratio(hull, box, shape)
                    max_overlap = max(max_overlap, ratio)
                if max_overlap < reject_ratio:
                    kept.append(det)
            return kept
        except Exception:
            return detections
