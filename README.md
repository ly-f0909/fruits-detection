# Augmented Vision

Local fruit & vegetable recognition inspired by [food-recognition](https://github.com/lannguyen0910/food-recognition): **YOLOv8s** detection (90 food classes, strong produce coverage) with a Streamlit UI for upload / camera capture and per-100g calorie references.

## # Augmented Vision — Getting Started

Fruit & vegetable recognition (YOLOv8s + Streamlit).

Model: lannguyen0910/food-recognition (YOLOv8s).

## Requirements

- Python 3.10+ (3.11/3.12 OK)

- macOS / Linux / Windows

- ~2GB free disk (venv + model)

- Network (first-time package + model download)

## 1. Get the code

# Option A — Git

git clone <REPO_URL>

cd augmented-vision   # or your folder name

# Option B — Zip

# Unzip the project folder, then cd into it

# Do NOT use a shared .venv from someone else

## 2. Create virtual environment

python3 -m venv .venv

# macOS / Linux

source .venv/bin/activate

# Windows (PowerShell)

.venv\Scripts\Activate.ps1

## 3. Install dependencies

pip install -r requirements.txt

## 4. Download model weights (~128 MB)

python scripts/download_[model.py](http://model.py)

# Expected file:

# models/food_[yolov8s.pt](http://yolov8s.pt)

# If Google Drive / gdown fails, ask the team lead for

# models/food_[yolov8s.pt](http://yolov8s.pt) and place it under models/

## 5. Run the web app

streamlit run [app.py](http://app.py)

# Browser should open [http://localhost:8501](http://localhost:8501)

# If not, open that URL manually.

## 6. (Optional) CLI test

python [cli.py](http://cli.py) path/to/photo.jpg --conf 0.25 -o tmp/result.jpg

## UI tips

- Sidebar: Upload image or Camera

- Default: "Fruits & vegetables only" is ON

- If nothing is detected: lower Confidence, or turn off produce-only

- First run may be slow (model load); later runs are faster

## Common issues

1. ModuleNotFoundError → activate .venv, then pip install -r requirements.txt

2. Model not found → run python scripts/download_[model.py](http://model.py)

3. Camera not working → use Upload image instead (browser permission)

4. Port 8501 busy → streamlit run [app.py](http://app.py) --server.port 8502

## Project layout

[app.py](http://app.py)                      # Streamlit UI (English)

[cli.py](http://cli.py)                      # Command-line inference

[detector.py](http://detector.py)                 # Detection + calorie lookup

scripts/download_[model.py](http://model.py)   # Download pretrained weights

models/food_[yolov8s.pt](http://yolov8s.pt)      # Weights (download, not in git)

requirements.txt

## Credits

- [lannguyen0910/food-recognition](https://github.com/lannguyen0910/food-recognition)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

