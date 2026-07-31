"""CLI helper: run food / produce detection on a local image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from detector import MODEL_PATH, calculate_calories, display_name, predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmented Vision food detector")
    parser.add_argument("image", type=Path, help="Path to an input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument(
        "--all-foods",
        action="store_true",
        help="Detect all food classes (default: fruits & vegetables only)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tmp/result.jpg"),
        help="Where to save the annotated image",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found: {MODEL_PATH}. Run: python scripts/download_model.py"
        )

    model = YOLO(str(MODEL_PATH))
    annotated, details = predict_image(
        args.image,
        model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        produce_only=not args.all_foods,
    )
    items = calculate_calories(details)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.output), bgr)

    if not items:
        print("No detections.")
    else:
        print("Detections:")
        for name, calories, conf in items:
            print(f"  - {display_name(name)}: {conf:.1%} | {calories} kcal/100g")
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
