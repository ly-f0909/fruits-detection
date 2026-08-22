"""Post-processing verification: hysteresis + temporal stability + hand filter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import AppConfig
from .detector import Detection
from .hand_filter import HandOcclusionFilter


@dataclass
class TrackVerificationState:
    """Per track_id history for anti-false-positive gating."""

    track_id: int
    label: str
    hit_count: int = 0
    hysteresis_confirmed: bool = False
    render_confirmed: bool = False
    missed_frames: int = 0
    # Rolling conf for debug / smoother handoff
    last_confidence: float = 0.0


@dataclass
class VerificationStats:
    """Counts from the last ``process`` call (for HUD/debug)."""

    raw: int = 0
    after_hand: int = 0
    after_hysteresis: int = 0
    confirmed: int = 0
    rejected_hand: int = 0
    rejected_conf: int = 0
    pending: int = 0


class VerificationPipeline:
    """
    Anti false-positive pipeline applied after YOLO ``track()``.

    Stages (in order):
      1. Hand occlusion filter (MediaPipe)
      2. Hysteresis confidence (high to confirm, low to maintain)
      3. Temporal stability (>= N consecutive frames, same class)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.hand_filter = HandOcclusionFilter(config)
        self._states: Dict[int, TrackVerificationState] = {}
        self.stats = VerificationStats()

    def close(self) -> None:
        self.hand_filter.close()

    def reset(self) -> None:
        self._states.clear()
        self.stats = VerificationStats()

    def _passes_hysteresis(self, state: TrackVerificationState, confidence: float) -> bool:
        if not state.hysteresis_confirmed:
            return confidence >= self.config.conf_high
        return confidence >= self.config.conf_maintain

    def process(self, frame_bgr: np.ndarray, detections: List[Detection]) -> List[Detection]:
        self.stats = VerificationStats(raw=len(detections))

        after_hand = self.hand_filter.filter(frame_bgr, detections)
        self.stats.after_hand = len(after_hand)
        self.stats.rejected_hand = self.stats.raw - self.stats.after_hand

        seen: set[int] = set()
        gated: List[Detection] = []

        for det in after_hand:
            seen.add(det.track_id)
            state = self._states.get(det.track_id)
            if state is None:
                state = TrackVerificationState(track_id=det.track_id, label=det.label)
                self._states[det.track_id] = state

            if not self._passes_hysteresis(state, det.confidence):
                self.stats.rejected_conf += 1
                continue

            if not state.hysteresis_confirmed and det.confidence >= self.config.conf_high:
                state.hysteresis_confirmed = True

            # Temporal: consecutive frames with consistent label
            if det.label == state.label:
                state.hit_count += 1
            else:
                state.label = det.label
                state.hit_count = 1
                state.render_confirmed = False

            state.missed_frames = 0
            state.last_confidence = det.confidence
            gated.append(det)

        self.stats.after_hysteresis = len(gated)

        confirmed: List[Detection] = []
        for det in gated:
            state = self._states[det.track_id]
            if state.hit_count >= self.config.temporal_min_hits:
                state.render_confirmed = True
                confirmed.append(det)
            else:
                self.stats.pending += 1

        # Age out tracks missing from ByteTrack output
        stale = self.config.verification_stale_frames
        for tid, state in list(self._states.items()):
            if tid not in seen:
                state.missed_frames += 1
                if state.missed_frames > stale:
                    del self._states[tid]
                else:
                    # Brief dropout: keep confirmed tracks visible via last box
                    if state.render_confirmed and state.hit_count >= self.config.temporal_min_hits:
                        self.stats.pending += 1

        self.stats.confirmed = len(confirmed)
        return confirmed

    @property
    def pending_tracks(self) -> int:
        return sum(
            1
            for s in self._states.values()
            if s.hysteresis_confirmed
            and not s.render_confirmed
            and s.hit_count < self.config.temporal_min_hits
        )
