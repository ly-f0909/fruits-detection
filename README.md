# Augmented Vision — Live AR Demo

Real-time multi-object **fruit & vegetable recognition** from the webcam, with a sci-fi **AR HUD** (corner reticles, leader lines, translucent info cards).

## Features

- Webcam capture with clear failure messages
- **63-class LVIS fruits & vegetables** model (pineapple, strawberry, grape, watermelon, kiwi, avocado, …)
- ByteTrack IDs + EMA box smoothing (anti-jitter)
- AR HUD / simple boxes / clean feed (`H` to cycle)
- FPS / target count / device overlay
- Screenshot to `screenshots/` (`S`)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py    # ~52 MB, once
python main.py --conf 0.20
```

### Useful flags

```bash
python main.py --conf 0.15                 # more sensitive
python main.py --fruits-only               # fruits only (no vegetables)
python main.py --device mps                # Apple Silicon
python main.py --imgsz 512                 # faster on CPU
```

### Keyboard

| Key | Action |
|-----|--------|
| `H` | Cycle mode: DETECTION → AR HUD → CLEAN |
| `S` | Save current frame to `screenshots/` |
| `Q` / `ESC` | Quit and release camera |

## Model

| Item | Value |
|------|-------|
| Default | `models/fruits_vegetables_yolov8m.pt` |
| Classes | **63** (LVIS fruits & vegetables subset) |
| Includes | apple, banana, pineapple, strawberry, grape, watermelon, kiwi, avocado, cherry, lemon, orange, peach, pear, blueberry, raspberry, papaya, coconut, … + vegetables |
| Source | [henningheyen/Fruits-And-Vegetables-Detection-Dataset](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset) |

Larger (slower/more accurate) weights also available under `models/lvis_fv/`:
- `yolo_fruits_and_vegetables_v2.pt` (large)
- `yolo_fruits_and_vegetables_v3.pt` (xlarge)

```bash
python main.py --model models/lvis_fv/yolo_fruits_and_vegetables_v2.pt --conf 0.20
```

## Project layout

```
.
├── main.py
├── live_ar/                # camera, detector, smoother, AR HUD
├── models/
│   └── fruits_vegetables_yolov8m.pt
├── scripts/download_model.py
└── screenshots/
```

## Credits

- [henningheyen/Fruits-And-Vegetables-Detection-Dataset](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
