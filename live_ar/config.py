"""Global configuration for the Augmented Vision live AR demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Preferred: LVIS fruits & vegetables YOLOv8m (63 classes, includes pineapple).
LVIS_FV_MODEL = ROOT / "models" / "fruits_vegetables_yolov8m.pt"
LVIS_FV_SOURCE = ROOT / "models" / "lvis_fv" / "yolo_fruits_and_vegetables_v1.pt"
FOOD_YOLO_PATH = ROOT / "models" / "food_yolov8s.pt"


class DisplayMode(Enum):
    """Overlay rendering modes toggled with the 'h' key."""

    DETECTION = 0
    AR_HUD = 1
    CLEAN = 2


# COCO fallback — only 5 produce classes (no pineapple).
COCO_PRODUCE_CLASS_IDS: Dict[int, str] = {
    46: "Banana",
    47: "Apple",
    49: "Orange",
    50: "Broccoli",
    51: "Carrot",
}

# Fruit-focused subset of the 63-class LVIS model (used when produce_only=True
# with --fruits-only). Vegetables remain available when produce_only keeps all
# model classes (default for the LVIS model).
FRUIT_NAME_KEYWORDS: Set[str] = {
    "almond",
    "apple",
    "apricot",
    "avocado",
    "banana",
    "blackberry",
    "blueberry",
    "cantaloup",
    "cantaloupe",
    "cherry",
    "clementine",
    "coconut",
    "date",
    "fig",
    "grape",
    "kiwi",
    "lemon",
    "lime",
    "mandarin",
    "melon",
    "orange",
    "papaya",
    "peach",
    "pear",
    "persimmon",
    "pineapple",
    "prune",
    "raspberry",
    "strawberry",
    "watermelon",
}

LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "apple": (60, 60, 255),
    "banana": (0, 220, 255),
    "orange": (0, 165, 255),
    "pineapple": (0, 200, 255),
    "strawberry": (80, 80, 255),
    "grape": (200, 80, 180),
    "watermelon": (60, 180, 60),
    "lemon": (0, 240, 240),
    "lime": (80, 230, 80),
    "kiwi fruit": (80, 180, 40),
    "mango": (0, 180, 255),
    "pear": (80, 200, 120),
    "peach": (140, 160, 255),
    "cherry": (40, 40, 220),
    "blueberry": (200, 120, 80),
    "avocado": (40, 160, 80),
    "broccoli": (80, 200, 80),
    "carrot": (0, 140, 255),
    "tomato": (40, 40, 230),
}

DEFAULT_ACCENT: Tuple[int, int, int] = (0, 255, 180)


def short_label(raw: str) -> str:
    """Turn LVIS-style names into short HUD labels (e.g. orange/orange fruit → Orange)."""
    primary = raw.split("/")[0].strip()
    # Drop trailing descriptors like "orange fruit" → keep first token group
    primary = primary.replace("_", " ")
    return primary[:1].upper() + primary[1:] if primary else raw


def default_model_path() -> str:
    """Prefer the 63-class LVIS fruits & vegetables YOLOv8m weights."""
    if LVIS_FV_MODEL.exists():
        return str(LVIS_FV_MODEL)
    if LVIS_FV_SOURCE.exists():
        return str(LVIS_FV_SOURCE)
    if FOOD_YOLO_PATH.exists():
        return str(FOOD_YOLO_PATH)
    return "yolov8n.pt"


def color_for_label(label: str) -> Tuple[int, int, int]:
    key = label.lower().split("/")[0].strip()
    if key in LABEL_COLORS:
        return LABEL_COLORS[key]
    for k, color in LABEL_COLORS.items():
        if k in key or key in k:
            return color
    return DEFAULT_ACCENT


@dataclass
class AppConfig:
    """Runtime knobs for camera, detector, tracker and HUD."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # 63-class LVIS YOLOv8m by default (apple…pineapple…watermelon…)
    model_name: str = field(default_factory=default_model_path)
    imgsz: int = 640
    conf_threshold: float = 0.20  # LVIS FV needs a lower threshold for live demos
    iou_threshold: float = 0.50
    device: str | None = None
    max_det: int = 30
    # When True and model is LVIS: keep all 63 classes.
    # When fruits_only=True: filter to fruit-like names only.
    produce_only: bool = True
    fruits_only: bool = False

    tracker_cfg: str = "bytetrack.yaml"
    ema_alpha: float = 0.35
    stale_frames: int = 15

    hud_alpha: float = 0.55
    reticle_len: int = 22
    reticle_thickness: int = 2
    card_width: int = 220
    card_height: int = 96

    screenshot_dir: str = "screenshots"
    window_name: str = "Augmented Vision | Live AR Demo"

    label_colors: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: dict(LABEL_COLORS)
    )
