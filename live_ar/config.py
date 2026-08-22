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

# Per-100g nutrition snippets shown on the AR HUD card.
NUTRITION_DATA: Dict[str, str] = {
    "apple": "~52 kcal/100g | High Fiber",
    "banana": "~89 kcal/100g | Rich in Vit B6",
    "grape": "~67 kcal/100g | Antioxidants",
    "orange": "~47 kcal/100g | Vitamin C",
    "lemon": "~29 kcal/100g | High Citric Acid",
    "strawberry": "~32 kcal/100g | Rich in Vit C",
    "watermelon": "~30 kcal/100g | Hydration",
    "pineapple": "~50 kcal/100g | Bromelain",
    "peach": "~39 kcal/100g | Vitamin A & C",
    "mango": "~60 kcal/100g | Rich in Fiber",
    "kiwi": "~61 kcal/100g | High Vit C",
    "tomato": "~18 kcal/100g | Lycopene",
    "carrot": "~41 kcal/100g | Beta-Carotene",
    "broccoli": "~34 kcal/100g | Sulforaphane",
    "cucumber": "~15 kcal/100g | Low Calorie",
}

DEFAULT_NUTRITION = "~50 kcal/100g | Organic Produce"


def get_nutrition_info(class_name: str) -> str:
    """
    Look up nutrition text for a detector label.

    Falls back to ``DEFAULT_NUTRITION`` when the class is unknown so HUD
    rendering never crashes on unseen LVIS labels.
    """
    key = class_name.lower().split("/")[0].strip().replace("_", " ")
    # Exact match, e.g. "apple"
    if key in NUTRITION_DATA:
        return NUTRITION_DATA[key]
    # First token match, e.g. "kiwi fruit" → "kiwi"
    first = key.split()[0] if key else ""
    if first in NUTRITION_DATA:
        return NUTRITION_DATA[first]
    # Substring match, e.g. "bell pepper" vs sparse keys
    for name, info in NUTRITION_DATA.items():
        if name in key or key in name:
            return info
    return NUTRITION_DATA.get(key, DEFAULT_NUTRITION)


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

    # Mac Continuity often takes index 0 (iPhone); built-in camera is usually 1.
    camera_index: int = 1
    frame_width: int = 960
    frame_height: int = 540

    # 63-class LVIS YOLOv8m by default (apple…pineapple…watermelon…)
    # 384 ≈ better FPS on CPU; raise to 416/640 when MPS (macOS 14+) is available.
    model_name: str = field(default_factory=default_model_path)
    imgsz: int = 384
    # YOLO inference floor (hysteresis maintain threshold); post-pipeline applies conf_high for new tracks.
    conf_threshold: float = 0.25
    conf_high: float = 0.40
    conf_maintain: float = 0.25
    iou_threshold: float = 0.50
    device: str | None = None
    max_det: int = 30
    produce_only: bool = True
    fruits_only: bool = False

    # Verification / anti-false-positive
    temporal_min_hits: int = 3
    verification_stale_frames: int = 8
    # Mild hand filter: only finger-sticks + clear holding hands (fruit-safe).
    enable_hand_filter: bool = True
    hand_overlap_reject: float = 0.75

    tracker_cfg: str = "bytetrack.yaml"
    ema_alpha: float = 0.35
    stale_frames: int = 15

    hud_alpha: float = 0.55
    reticle_len: int = 18
    reticle_thickness: int = 2
    card_width: int = 220
    card_height: int = 58

    screenshot_dir: str = "screenshots"
    window_name: str = "Augmented Vision | Live AR Demo"

    label_colors: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: dict(LABEL_COLORS)
    )
