"""Augmented Vision — fruit & vegetable recognition (YOLOv8 + Streamlit)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from detector import (
    TMP_DIR,
    Config,
    calculate_calories,
    display_name,
    load_model,
    predict_image,
)

st.set_page_config(
    page_title="Augmented Vision | Food Recognition",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
      .result-card {
        background: linear-gradient(145deg, #f7faf5 0%, #eef5ea 100%);
        border: 1px solid #d5e5cf;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
      }
      .muted { color: #5b6b57; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


def save_upload(image: Image.Image, filename: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / filename
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, format="JPEG", quality=92)
    return path


def main() -> None:
    st.title("Augmented Vision")
    st.caption(
        "Fruit & vegetable recognition with YOLOv8s · image upload / camera · calorie reference"
    )

    with st.sidebar:
        st.header("Input")
        source = st.radio("Image source", ("Upload image", "Camera"), index=0)
        conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
        iou = st.slider("IoU threshold", 0.10, 0.95, 0.45, 0.05)
        produce_only = st.checkbox("Fruits & vegetables only", value=True)
        st.markdown("---")
        with st.expander(f"Detectable classes ({len(Config.CLASSES)})", expanded=False):
            names = (
                sorted(Config.PRODUCE_CLASSES)
                if produce_only
                else Config.CLASSES
            )
            for name in names:
                st.write(f"- {display_name(name)}")
        st.markdown(
            '<p class="muted">Weights from '
            '<a href="https://github.com/lannguyen0910/food-recognition" target="_blank">'
            "lannguyen0910/food-recognition</a> (YOLOv8s, mAP@0.5 ≈ 0.96)</p>",
            unsafe_allow_html=True,
        )

    image: Image.Image | None = None
    image_path: Path | None = None

    if source == "Upload image":
        uploaded = st.sidebar.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )
        if uploaded is not None:
            image = Image.open(uploaded)
            image_path = save_upload(image, "upload.jpg")
    else:
        camera = st.sidebar.camera_input("Take a photo")
        if camera is not None:
            image = Image.open(camera)
            image_path = save_upload(image, "camera.jpg")

    if image is None or image_path is None:
        st.info("Upload an image or capture one with the camera to start recognition.")
        cols = st.columns(3)
        samples = [
            ("Fruits", "Apple, banana, strawberry, orange, mango…"),
            ("Vegetables", "Broccoli, carrot, tomato, cucumber, pepper…"),
            ("Nutrition", "Each detection includes kcal / 100g reference"),
        ]
        for col, (title, desc) in zip(cols, samples):
            with col:
                st.markdown(f"**{title}**")
                st.write(desc)
        return

    try:
        model = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    with st.spinner("Detecting…"):
        annotated, details = predict_image(
            image_path,
            model,
            conf_threshold=conf,
            iou_threshold=iou,
            produce_only=produce_only,
        )
        items = calculate_calories(details)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)

    st.subheader("Results")
    if not items:
        st.warning(
            "No items detected. Try lowering the confidence threshold, "
            "disabling produce-only mode, or using a clearer photo."
        )
        return

    for name, calories, confidence in items:
        st.markdown(
            f"""
            <div class="result-card">
              <strong>{display_name(name)}</strong><br/>
              <span class="muted">Confidence {confidence:.1%} · {calories} kcal / 100g (ref.)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    total_ref = sum(c for _, c, _ in items)
    st.caption(
        f"Found {len(items)} class(es). Calories are per-100g references "
        f"(not portion estimates). Sum of listed refs: {total_ref} kcal/100g."
    )


if __name__ == "__main__":
    main()
