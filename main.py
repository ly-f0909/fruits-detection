#!/usr/bin/env python3
"""
Augmented Vision — Live AR Demo
================================
Real-time multi-object fruit/vegetable recognition from the webcam with a
sci-fi AR HUD overlay (corner reticles, leader lines, translucent info cards).

Usage
-----
    python main.py
    python main.py --camera 1 --conf 0.4
    python main.py --device cpu --imgsz 320

Keys
----
    h / H   cycle display mode  (DETECTION → AR HUD → CLEAN)
    s / S   save screenshot to screenshots/
    q / ESC quit and release the camera
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from live_ar.ar_renderer import render_detections
from live_ar.camera import CameraError, WebcamCapture
from live_ar.config import AppConfig, DisplayMode, default_model_path
from live_ar.detector import ProduceDetector
from live_ar.hud_panel import draw_perf_panel
from live_ar.smoother import EMABoxSmoother


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augmented Vision — real-time produce AR HUD demo",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument(
        "--model",
        type=str,
        default=default_model_path(),
        help="Ultralytics weights (default: 63-class LVIS fruits & vegetables)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold (0.15–0.25 recommended for LVIS model)",
    )
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: cpu | mps | 0 (CUDA). Default = auto",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.35,
        help="EMA smoothing factor (0.05–1.0, lower = smoother)",
    )
    parser.add_argument(
        "--fruits-only",
        action="store_true",
        help="Keep fruit classes only (drop vegetables)",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Do not restrict class filter (same as full 63 on LVIS model)",
    )
    return parser.parse_args()


def save_screenshot(frame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"ar_capture_{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def cycle_mode(mode: DisplayMode) -> DisplayMode:
    values = list(DisplayMode)
    return values[(mode.value + 1) % len(values)]


def main() -> int:
    args = parse_args()
    config = AppConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        model_name=args.model,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        ema_alpha=args.ema,
        produce_only=not args.all_classes,
        fruits_only=args.fruits_only,
    )

    screenshot_dir = Path(config.screenshot_dir)
    mode = DisplayMode.AR_HUD

    print("=" * 56)
    print("  Augmented Vision — Live AR Demo")
    print("=" * 56)
    print(f"  Camera #{config.camera_index}  |  model={config.model_name}")
    print("  Keys: [H] mode  [S] screenshot  [Q]/ESC] quit")
    print("=" * 56)

    try:
        detector = ProduceDetector(config)
    except Exception as exc:  # noqa: BLE001 — surface model download / load errors
        print(f"[ERROR] Failed to load model '{config.model_name}': {exc}", file=sys.stderr)
        print(
            "Hint: ensure network access on first run so Ultralytics can download weights.",
            file=sys.stderr,
        )
        return 1

    print(f"[INFO] Inference device: {detector.device_label}")
    smoother = EMABoxSmoother(alpha=config.ema_alpha, stale_frames=config.stale_frames)

    try:
        cam = WebcamCapture(config.camera_index, config.frame_width, config.frame_height)
        cam.open()
    except CameraError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    fps_ema = 0.0
    last_t = time.perf_counter()
    status_msg = ""
    status_ttl = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[WARN] Empty frame from camera — retrying…")
                time.sleep(0.02)
                continue

            # --- detect + smooth ---
            raw_dets = detector.infer(frame)
            dets = smoother.update(raw_dets)

            # --- render ---
            canvas = frame.copy()
            canvas = render_detections(canvas, dets, mode, config)

            now = time.perf_counter()
            dt = max(now - last_t, 1e-6)
            last_t = now
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema <= 1e-3 else (0.90 * fps_ema + 0.10 * inst_fps)

            canvas = draw_perf_panel(
                canvas,
                fps=fps_ema,
                num_targets=len(dets),
                device_label=detector.device_label,
                mode=mode,
            )

            if len(dets) == 0 and mode is not DisplayMode.CLEAN:
                cv2.putText(
                    canvas,
                    "No produce detected — try Apple/Banana/Pineapple  (or --conf 0.15)",
                    (12, canvas.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

            if status_ttl > 0 and status_msg:
                cv2.putText(
                    canvas,
                    status_msg,
                    (12, canvas.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 180),
                    2,
                    cv2.LINE_AA,
                )
                status_ttl -= 1

            cv2.imshow(config.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break
            if key in (ord("h"), ord("H")):
                mode = cycle_mode(mode)
                status_msg = f"Mode → {mode.name}"
                status_ttl = 45
            if key in (ord("s"), ord("S")):
                path = save_screenshot(canvas, screenshot_dir)
                status_msg = f"Saved {path.name}"
                status_ttl = 60
                print(f"[INFO] Screenshot → {path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Bye.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
