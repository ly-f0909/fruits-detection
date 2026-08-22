"""Hand occlusion filter for produce false-positive suppression.

Strategy (in order):
  1. MediaPipe Tasks HandLandmarker with forced CPU delegate
  2. Legacy MediaPipe Solutions Hands (older wheels)
  3. Lightweight OpenCV skin-color heuristic (always available)

On macOS, MediaPipe 1.x may FATAL-abort via Metal (`DrishtiMetalHelper`)
even when CPU is requested. We smoke-test Tasks init in a **child process**
first; if that aborts, we skip MediaPipe and use the OpenCV fallback so the
live demo never crashes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

# Must be set before MediaPipe graphs try to bind Metal helpers.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import numpy as np

from .config import AppConfig
from .detector import Detection
from .geometry import landmarks_to_hull, polygon_box_overlap_ratio

WARN_MSG = "[WARN] MediaPipe hands unavailable — using OpenCV skin fallback"
SKIP_MSG = "[WARN] MediaPipe hands unavailable, skipping hand filter"

ROOT = Path(__file__).resolve().parents[1]
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


class _HandBackend(Protocol):
    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        ...

    def close(self) -> None:
        ...


class _SolutionsHandsBackend:
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
    def __init__(self, landmarker: Any, *, video_mode: bool) -> None:
        self._landmarker = landmarker
        self._video_mode = video_mode
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video_mode:
            self._timestamp_ms = int(time.monotonic() * 1000)
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return []
        return [
            landmarks_to_hull(hand_lms, w, h) for hand_lms in result.hand_landmarks
        ]

    def close(self) -> None:
        self._landmarker.close()


class _OpenCVSkinHandBackend:
    """
    Fast CPU-only fallback: YCrCb skin mask → contours as hand hulls.

    Less precise than MediaPipe, but safe on macOS and enough to reject
    finger/palm false positives that heavily overlap produce boxes.
    """

    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        scale = 0.5 if max(h, w) > 640 else 1.0
        small = (
            cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else frame_bgr
        )
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (small.shape[0] * small.shape[1]) * 0.01
        hulls: List[np.ndarray] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            hull = cv2.convexHull(cnt)
            if scale < 1.0:
                hull = (hull.astype(np.float32) / scale).astype(np.int32)
            hulls.append(hull)
        return hulls

    def close(self) -> None:
        return


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


def _tasks_cpu_smoke_ok(model_path: Path) -> bool:
    """Probe HandLandmarker CPU init in a child process (Metal abort isolation)."""
    script = f"""
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
opts = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path={str(model_path)!r},
        delegate=BaseOptions.Delegate.CPU,
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
)
lm = HandLandmarker.create_from_options(opts)
lm.close()
print("HAND_SMOKE_OK")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ, "MEDIAPIPE_DISABLE_GPU": "1"},
        )
        return result.returncode == 0 and "HAND_SMOKE_OK" in (result.stdout or "")
    except Exception:
        return False


def _load_solutions_backend(config: AppConfig) -> _HandBackend:
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
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    model_path = _ensure_hand_model()
    if not _tasks_cpu_smoke_ok(model_path):
        raise RuntimeError(
            "HandLandmarker init aborts on this Mac (MediaPipe Metal bug); "
            "use OpenCV skin fallback"
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = HandLandmarker.create_from_options(options)
    return _TasksHandLandmarkerBackend(landmarker, video_mode=False)


def _create_hand_backend(config: AppConfig) -> Optional[_HandBackend]:
    # macOS 13 + MediaPipe Tasks often Metal-aborts; child smoke test also costs
    # ~10s every launch. Prefer fast OpenCV skin on Darwin for live demos.
    if sys.platform == "darwin":
        loaders = (("opencv.skin", lambda _c: _OpenCVSkinHandBackend()),)
        print("[INFO] macOS: using OpenCV skin hand filter (skip MediaPipe)")
    else:
        loaders = (
            ("mediapipe.tasks.CPU", _load_tasks_backend),
            ("mediapipe.python.solutions", _load_solutions_backend),
            ("mp.solutions.hands", _load_legacy_solutions_backend),
            ("opencv.skin", lambda _c: _OpenCVSkinHandBackend()),
        )
    for label, loader in loaders:
        try:
            backend = loader(config)
            print(f"[INFO] Hand occlusion filter enabled ({label})")
            return backend
        except (ImportError, ModuleNotFoundError, AttributeError, OSError, RuntimeError, ValueError) as exc:
            print(f"[INFO] Hand backend '{label}' skipped: {exc}")
            continue
        except Exception as exc:
            print(f"[INFO] Hand backend '{label}' skipped: {exc}")
            continue
    return None


class HandOcclusionFilter:
    """
    Hand / palm occlusion filter.

    Prefers MediaPipe when safe; otherwise falls back to OpenCV skin detection
    so the live AR demo never crashes on macOS Metal aborts.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = bool(getattr(config, "enable_hand_filter", True))
        self._backend: Optional[_HandBackend] = None

        if not self.enabled:
            return

        try:
            self._backend = _create_hand_backend(config)
            if self._backend is None:
                print(SKIP_MSG)
                self.enabled = False
            elif isinstance(self._backend, _OpenCVSkinHandBackend):
                print(WARN_MSG)
        except Exception:
            print(SKIP_MSG)
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
            # Fruit-safe: only drop elongated skin sticks (empty fingers / palm edge).
            # Do NOT reject produce that is being held — that kills handheld fruit demos.
            use_skin = isinstance(self._backend, _OpenCVSkinHandBackend)
            if not use_skin:
                # MediaPipe path (non-Mac): classic overlap reject vs true hand hulls.
                hulls = self._hand_hulls(frame_bgr)
                if not hulls:
                    return detections
                reject_ratio = float(getattr(self.config, "hand_overlap_reject", 0.75))
                shape = frame_bgr.shape
                kept: List[Detection] = []
                for det in detections:
                    box = (det.x1, det.y1, det.x2, det.y2)
                    max_overlap = 0.0
                    for hull in hulls:
                        ratio = polygon_box_overlap_ratio(hull, box, shape)
                        max_overlap = max(max_overlap, ratio)
                    if max_overlap < reject_ratio:
                        kept.append(det)
                return kept

            skin_mask = _skin_mask(frame_bgr)
            kept = [
                det
                for det in detections
                if not _box_looks_like_finger((det.x1, det.y1, det.x2, det.y2), skin_mask)
            ]
            return kept
        except Exception:
            return detections


def _skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return cv2.medianBlur(mask, 5)


def _box_looks_like_finger(
    box: Tuple[float, float, float, float],
    skin_mask: np.ndarray,
) -> bool:
    """
    Reject only very elongated skin sticks (fingers / sideways palm).

    Round produce (apple/orange/tomato) has aspect ≈ 1 → never rejected here.
    Banana is elongated but usually not skin-pink enough to hit fill≥0.55.
    """
    h, w = skin_mask.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return False

    bw, bh = x2 - x1, y2 - y1
    aspect = bw / float(bh)
    # Must be clearly elongated (fingers), not a fruit disc.
    if 0.55 < aspect < 1.80:
        return False

    roi = skin_mask[y1:y2, x1:x2]
    fill = float(np.count_nonzero(roi)) / float(roi.size)
    return fill >= 0.55


def _hull_looks_like_holding_hand(
    hull: np.ndarray,
    box: Tuple[float, float, float, float],
    frame_shape: tuple,
) -> bool:
    """True if skin blob extends meaningfully outside the produce box."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    mask_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_poly, hull.astype(np.int32), 255)
    poly_area = float(np.count_nonzero(mask_poly))
    if poly_area < 1.0:
        return False

    mask_box = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask_box, (x1, y1), (x2, y2), 255, thickness=-1)
    inside = float(np.count_nonzero(cv2.bitwise_and(mask_poly, mask_box)))
    outside_ratio = (poly_area - inside) / poly_area
    # Holding hand only: ≥50% of skin blob outside the fruit box.
    return outside_ratio >= 0.50
