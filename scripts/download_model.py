"""Download food-recognition YOLOv8s weights into models/."""

from __future__ import annotations

from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "food_yolov8s.pt"
# From https://github.com/lannguyen0910/food-recognition theseus/utilities/download.py
DRIVE_ID = "1f2kOOyCQ8aHzSHPH8jf9Z6cT4ai-yqmx"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 1_000_000:
        print(f"Already present: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    print(f"Downloading YOLOv8s food weights to {OUT} …")
    gdown.download(url, str(OUT), quiet=False)
    print(f"Done: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
