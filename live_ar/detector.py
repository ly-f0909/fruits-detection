"""YOLOv8 produce detector with ByteTrack multi-object tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO

from .config import (
    COCO_PRODUCE_CLASS_IDS,
    FRUIT_NAME_KEYWORDS,
    AppConfig,
    short_label,
)


@dataclass
class Detection:
    """One tracked produce instance in image space."""

    track_id: int
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def top_center(self) -> tuple[float, float]:
        return self.cx, self.y1


def resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _names_to_id_map(names: dict | list) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def _is_fruit_name(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in FRUIT_NAME_KEYWORDS)


class ProduceDetector:
    """
    Real-time fruit/vegetable detector.

    Default weights: LVIS Fruits & Vegetables YOLOv8m (63 classes), including
    pineapple, strawberry, grape, watermelon, kiwi, avocado, cherry, …
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        model_path = config.model_name

        if not Path(model_path).exists() and model_path not in {"yolov8n.pt", "yolo11n.pt"}:
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run: python scripts/download_model.py\n"
                "Or:  python main.py --model yolov8n.pt"
            )

        self.model = YOLO(model_path)
        self.id_to_label = _names_to_id_map(self.model.names)

        # Build allowed class set.
        if len(self.id_to_label) >= 50:
            # LVIS 63-class model: use all classes, optionally fruits-only.
            if config.fruits_only:
                self.allowed_ids = {
                    cid: name
                    for cid, name in self.id_to_label.items()
                    if _is_fruit_name(name)
                }
            else:
                self.allowed_ids = dict(self.id_to_label)
        elif "Pineapple" in self.id_to_label.values() or "pineapple" in {
            v.lower() for v in self.id_to_label.values()
        }:
            self.allowed_ids = dict(self.id_to_label)
            if config.fruits_only:
                self.allowed_ids = {
                    cid: name
                    for cid, name in self.allowed_ids.items()
                    if _is_fruit_name(name)
                }
        elif "yolov8n" in str(model_path).lower() or "yolo11n" in str(model_path).lower():
            self.allowed_ids = dict(COCO_PRODUCE_CLASS_IDS)
        else:
            self.allowed_ids = dict(self.id_to_label)

        if config.produce_only or config.fruits_only:
            self.class_filter: Optional[List[int]] = sorted(self.allowed_ids.keys())
        else:
            self.class_filter = None

        n = len(self.allowed_ids)
        print(f"[INFO] Model={model_path}")
        print(f"[INFO] Classes in filter: {n}")
        has_pine = any("pineapple" in v.lower() for v in self.allowed_ids.values())
        print(f"[INFO] Pineapple supported: {'YES' if has_pine else 'NO'}")

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        track_kwargs = dict(
            source=frame_bgr,
            persist=True,
            tracker=self.config.tracker_cfg,
            conf=min(self.config.conf_threshold, self.config.conf_maintain),
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            device=self.device,
            max_det=self.config.max_det,
            verbose=False,
        )
        if self.class_filter is not None:
            track_kwargs["classes"] = self.class_filter

        results = self.model.track(**track_kwargs)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(xyxy), dtype=int)

        detections: List[Detection] = []
        for i in range(len(xyxy)):
            class_id = int(cls_ids[i])
            raw = self.allowed_ids.get(class_id) or self.id_to_label.get(class_id)
            if raw is None:
                continue
            if self.class_filter is not None and class_id not in self.allowed_ids:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            detections.append(
                Detection(
                    track_id=int(track_ids[i]),
                    class_id=class_id,
                    label=short_label(raw),
                    confidence=float(confs[i]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    @property
    def device_label(self) -> str:
        if self.device in ("0", "cuda", "cuda:0"):
            return "CUDA"
        if self.device == "mps":
            return "MPS"
        return "CPU"
