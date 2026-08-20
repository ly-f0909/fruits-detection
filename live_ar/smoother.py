"""Frame-to-frame EMA smoothing for bounding boxes and anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .detector import Detection


@dataclass
class SmoothedTrack:
    """EMA state for one track_id."""

    track_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    missed: int = 0

    def as_detection(self) -> Detection:
        return Detection(
            track_id=self.track_id,
            class_id=-1,
            label=self.label,
            confidence=self.confidence,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )


class EMABoxSmoother:
    """
    Exponential moving average on xyxy boxes.

    ByteTrack already stabilizes identities; EMA removes residual jitter so
    reticles and HUD leader-lines glide instead of flicker.
    """

    def __init__(self, alpha: float = 0.35, stale_frames: int = 15) -> None:
        self.alpha = float(np_clip(alpha, 0.05, 1.0))
        self.stale_frames = stale_frames
        self._tracks: Dict[int, SmoothedTrack] = {}

    def update(self, detections: List[Detection]) -> List[Detection]:
        seen: set[int] = set()
        a = self.alpha

        for det in detections:
            seen.add(det.track_id)
            prev = self._tracks.get(det.track_id)
            if prev is None:
                self._tracks[det.track_id] = SmoothedTrack(
                    track_id=det.track_id,
                    label=det.label,
                    confidence=det.confidence,
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    missed=0,
                )
                continue

            # EMA blend: smooth = α * raw + (1-α) * previous
            prev.x1 = a * det.x1 + (1.0 - a) * prev.x1
            prev.y1 = a * det.y1 + (1.0 - a) * prev.y1
            prev.x2 = a * det.x2 + (1.0 - a) * prev.x2
            prev.y2 = a * det.y2 + (1.0 - a) * prev.y2
            prev.confidence = a * det.confidence + (1.0 - a) * prev.confidence
            prev.label = det.label
            prev.missed = 0

        # Age out disappeared tracks
        for tid, track in list(self._tracks.items()):
            if tid not in seen:
                track.missed += 1
                if track.missed > self.stale_frames:
                    del self._tracks[tid]

        return [t.as_detection() for t in self._tracks.values() if t.missed == 0]


def np_clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
