"""Food / produce detection helpers powered by food-recognition YOLOv8s."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "food_yolov8s.pt"
TMP_DIR = ROOT / "tmp"

# Class order matches models/food_yolov8s.pt (lannguyen0910/food-recognition).
CLASS_NAMES = [
    "hot-dog",
    "Apple",
    "Artichoke",
    "Asparagus",
    "Bagel",
    "Baked-goods",
    "Banana",
    "Beer",
    "Bell-pepper",
    "Bread",
    "Broccoli",
    "Burrito",
    "Cabbage",
    "Cake",
    "Candy",
    "Cantaloupe",
    "Carrot",
    "Common-fig",
    "Cookie",
    "Dessert",
    "French-fries",
    "Grape",
    "Guacamole",
    "Hot-dog",
    "Ice-cream",
    "Muffin",
    "Orange",
    "Pancake",
    "Pear",
    "Popcorn",
    "Pretzel",
    "Strawberry",
    "Tomato",
    "Waffle",
    "food-drinks",
    "Cheese",
    "Cocktail",
    "Coffee",
    "Cooking-spray",
    "Crab",
    "Croissant",
    "Cucumber",
    "Doughnut",
    "Egg",
    "Fruit",
    "Grapefruit",
    "Hamburger",
    "Honeycomb",
    "Juice",
    "Lemon",
    "Lobster",
    "Mango",
    "Milk",
    "Mushroom",
    "Oyster",
    "Pasta",
    "Pastry",
    "Peach",
    "Pineapple",
    "Pizza",
    "Pomegranate",
    "Potato",
    "Pumpkin",
    "Radish",
    "Salad",
    "food",
    "Sandwich",
    "Shrimp",
    "Squash",
    "Squid",
    "Submarine-sandwich",
    "Sushi",
    "Taco",
    "Tart",
    "Tea",
    "Vegetable",
    "Watermelon",
    "Wine",
    "Winter-melon",
    "Zucchini",
    "Banh_mi",
    "Banh_trang_tron",
    "Banh_xeo",
    "Bun_bo_Hue",
    "Bun_dau",
    "Com_tam",
    "Goi_cuon",
    "Pho",
    "Hu_tieu",
    "Xoi",
]

# Approximate kcal per 100g (reference values).
CALORIES_DICT: dict[str, int] = {
    "Apple": 52,
    "Artichoke": 47,
    "Asparagus": 20,
    "Bagel": 257,
    "Baked-goods": 300,
    "Banana": 89,
    "Beer": 43,
    "Bell-pepper": 20,
    "Bread": 265,
    "Broccoli": 34,
    "Burrito": 206,
    "Cabbage": 25,
    "Cake": 350,
    "Candy": 380,
    "Cantaloupe": 34,
    "Carrot": 41,
    "Cheese": 402,
    "Cocktail": 120,
    "Coffee": 2,
    "Common-fig": 74,
    "Cookie": 480,
    "Cooking-spray": 0,
    "Crab": 97,
    "Croissant": 406,
    "Cucumber": 16,
    "Dessert": 300,
    "Doughnut": 421,
    "Egg": 155,
    "French-fries": 312,
    "Fruit": 50,
    "Grape": 69,
    "Grapefruit": 42,
    "Guacamole": 160,
    "Hamburger": 295,
    "Honeycomb": 304,
    "Hot-dog": 290,
    "hot-dog": 290,
    "Ice-cream": 207,
    "Juice": 45,
    "Lemon": 29,
    "Lobster": 89,
    "Mango": 60,
    "Milk": 42,
    "Muffin": 377,
    "Mushroom": 22,
    "Orange": 47,
    "Oyster": 68,
    "Pancake": 227,
    "Pasta": 131,
    "Pastry": 400,
    "Peach": 39,
    "Pear": 57,
    "Pineapple": 50,
    "Pizza": 266,
    "Pomegranate": 83,
    "Popcorn": 375,
    "Potato": 77,
    "Pretzel": 380,
    "Pumpkin": 26,
    "Radish": 16,
    "Salad": 20,
    "Sandwich": 250,
    "Shrimp": 99,
    "Squash": 16,
    "Squid": 92,
    "Strawberry": 32,
    "Submarine-sandwich": 240,
    "Sushi": 150,
    "Taco": 226,
    "Tart": 320,
    "Tea": 1,
    "Tomato": 18,
    "Vegetable": 30,
    "Waffle": 291,
    "Watermelon": 30,
    "Wine": 83,
    "Winter-melon": 13,
    "Zucchini": 17,
    "food": 150,
    "food-drinks": 80,
    "Banh_mi": 250,
    "Banh_trang_tron": 180,
    "Banh_xeo": 220,
    "Bun_bo_Hue": 120,
    "Bun_dau": 180,
    "Com_tam": 160,
    "Goi_cuon": 100,
    "Pho": 90,
    "Hu_tieu": 110,
    "Xoi": 200,
}

PRODUCE_CLASSES = {
    "Apple",
    "Artichoke",
    "Asparagus",
    "Banana",
    "Bell-pepper",
    "Broccoli",
    "Cabbage",
    "Cantaloupe",
    "Carrot",
    "Common-fig",
    "Cucumber",
    "Fruit",
    "Grape",
    "Grapefruit",
    "Lemon",
    "Mango",
    "Mushroom",
    "Orange",
    "Peach",
    "Pear",
    "Pineapple",
    "Pomegranate",
    "Potato",
    "Pumpkin",
    "Radish",
    "Salad",
    "Squash",
    "Strawberry",
    "Tomato",
    "Vegetable",
    "Watermelon",
    "Winter-melon",
    "Zucchini",
}


class Config:
    """Detection config for food-recognition YOLOv8s (90 classes)."""

    CLASSES = CLASS_NAMES
    CALORIES_DICT = CALORIES_DICT
    PRODUCE_CLASSES = PRODUCE_CLASSES


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ").replace("-", " ")


@st.cache_resource
def load_model(model_path: str | Path | None = None) -> YOLO:
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run `python scripts/download_model.py` "
            "or download YOLOv8s weights from "
            "https://github.com/lannguyen0910/food-recognition"
        )
    return YOLO(str(path))


def resolve_class_name(model: YOLO, class_id: int) -> str | None:
    names = model.names
    if isinstance(names, dict):
        name = names.get(class_id)
    else:
        name = names[class_id] if 0 <= class_id < len(names) else None
    if name is None and 0 <= class_id < len(CLASS_NAMES):
        name = CLASS_NAMES[class_id]
    return name


def predict_image(
    image_source: str | Path | np.ndarray,
    model: YOLO,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    produce_only: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run YOLO detection and draw boxes. Returns RGB image + detection list."""
    results = model.predict(
        source=image_source,
        imgsz=640,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    if isinstance(image_source, (str, Path)):
        image = cv2.imread(str(image_source))
        if image is None:
            raise ValueError(f"Cannot read image: {image_source}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.asarray(image_source).copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    detection_details: list[dict[str, Any]] = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image, detection_details

    for row in boxes.data:
        x1, y1, x2, y2, confidence, class_id = row.cpu().numpy()
        class_name = resolve_class_name(model, int(class_id))
        if class_name is None:
            continue
        if produce_only and class_name not in PRODUCE_CLASSES:
            continue

        label = f"{display_name(class_name)} {confidence:.0%}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (46, 160, 67), 2)
        text_y = max(20, int(y1) - 8)
        cv2.putText(
            image,
            label,
            (int(x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (46, 160, 67),
            2,
            cv2.LINE_AA,
        )
        detection_details.append(
            {
                "class": class_name,
                "top_confidence": float(confidence),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            }
        )

    return image, detection_details


def calculate_calories(
    detection_details: list[dict[str, Any]],
) -> list[tuple[str, int, float]]:
    """Keep highest-confidence hit per class; return (name, kcal/100g, conf)."""
    unique_items: dict[str, dict[str, float | int]] = {}
    for det in detection_details:
        item = det["class"]
        confidence = float(det["top_confidence"])
        if item not in unique_items or confidence > float(unique_items[item]["confidence"]):
            unique_items[item] = {
                "calories": CALORIES_DICT.get(item, 0),
                "confidence": confidence,
            }

    detected_items = [
        (item, int(data["calories"]), float(data["confidence"]))
        for item, data in unique_items.items()
    ]
    detected_items.sort(key=lambda x: x[2], reverse=True)
    return detected_items
