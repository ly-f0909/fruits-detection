"""Download the default fruit/vegetable detection weights."""

from __future__ import annotations

import shutil
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LVIS_DIR = MODELS / "lvis_fv"
TARGET = MODELS / "fruits_vegetables_yolov8m.pt"

# Google Drive folder from:
# https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset
FOLDER_ID = "1I4mtQK11C3p41pO9raR0trgPVj0eQ2yb"
# YOLOv8m (v1) — best speed/coverage tradeoff for live demos (~52 MB)
V1_FILE_ID = "1XALsPwM4850LkmqsXIfmGhPj8s_nhuiK"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    LVIS_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET.exists() and TARGET.stat().st_size > 1_000_000:
        print(f"Already present: {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
        return

    v1 = LVIS_DIR / "yolo_fruits_and_vegetables_v1.pt"
    if not v1.exists() or v1.stat().st_size < 1_000_000:
        print("Downloading LVIS Fruits & Vegetables YOLOv8m (63 classes)…")
        gdown.download(
            f"https://drive.google.com/uc?id={V1_FILE_ID}",
            str(v1),
            quiet=False,
        )

    shutil.copy2(v1, TARGET)
    print(f"Ready: {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
    print("Classes: apple, banana, pineapple, strawberry, grape, watermelon, kiwi, … (63 total)")


if __name__ == "__main__":
    main()
