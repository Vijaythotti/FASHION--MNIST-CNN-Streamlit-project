from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_mnist_utils import (
    LEGACY_MODEL_PATH,
    MODEL_PATH,
    load_cnn_model,
    load_image,
    predict_image,
    top_k_predictions,
)


MODE_MAP = {
    "Auto Detect Mode": "auto",
    "Real Photo Mode": "real_photo",
    "Fashion-MNIST Mode": "fashion_mnist",
}

BACKGROUND_MAP = {
    "Auto": "auto",
    "Light Background": "light",
    "Dark Background": "dark",
}


def render_missing_model() -> None:
    st.error("Model file is missing.")
    st.info(
        "Train the CNN with `python3 train_model.py`. The app expects "
        "`models/fashion_mnist_cnn.keras`, or a fallback `models/fashion_mnist_softmax.npz`."
    )


def render_prediction() -> None:
    st.set_page_config(page_title="Fashion-MNIST CNN", page_icon="👕", layout="wide")
    st.title("Fashion-MNIST CNN Classifier")

    if not MODEL_PATH.exists() and not LEGACY_MODEL_PATH.exists():
        render_missing_model()
        return

    try:
        model = load_cnn_model()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if getattr(model, "model_type", "cnn") == "legacy_softmax":
        st.warning(
            "Using the legacy NumPy softmax model because the CNN model has not been trained yet. "
            "For best results, train `models/fashion_mnist_cnn.keras` in a Python 3.10-3.12 environment."
        )

    with st.sidebar:
        st.header("Prediction Settings")
        mode_label = st.radio("Prediction mode", list(MODE_MAP.keys()), index=0)
        background_label = st.radio("Background", list(BACKGROUND_MAP.keys()), index=0)
        cleanup_threshold = st.slider("Cleanup threshold", 0.01, 0.60, 0.18, 0.01)
        use_otsu = st.checkbox("Use OpenCV Otsu thresholding", value=True)

    uploaded_file = st.file_uploader("Upload a fashion image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is None:
        st.info("Upload a Fashion-MNIST-like image or a real fashion product photo.")
        return

    try:
        image = load_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        probabilities, preprocess_result = predict_image(
            model=model,
            image=image,
            mode=MODE_MAP[mode_label],
            background=BACKGROUND_MAP[background_label],
            cleanup_threshold=cleanup_threshold,
            use_otsu=use_otsu,
        )
    except Exception as exc:
        st.error("Preprocessing or prediction failed. Try another image or adjust the settings.")
        st.exception(exc)
        return

    top3 = top_k_predictions(probabilities, k=3)
    predicted_class, confidence = top3[0]

    original_col, processed_col = st.columns(2)
    with original_col:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with processed_col:
        st.subheader("Processed 28x28 CNN Input")
        st.image(
            preprocess_result.processed_image.resize((224, 224)),
            caption="This is the exact image tensor fed into the CNN.",
            width=224,
        )

    st.divider()
    metric_col, table_col = st.columns([1, 2])
    with metric_col:
        st.metric("Predicted class", predicted_class)
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    with table_col:
        top3_df = pd.DataFrame(
            {
                "Class": [class_name for class_name, _probability in top3],
                "Probability": [probability for _class_name, probability in top3],
                "Percent": [f"{probability * 100:.2f}%" for _class_name, probability in top3],
            }
        )
        st.dataframe(top3_df, hide_index=True, use_container_width=True)

    chart_df = pd.DataFrame(
        {
            "Class": [class_name for class_name, _probability in top3],
            "Probability": [probability for _class_name, probability in top3],
        }
    )
    st.bar_chart(chart_df, x="Class", y="Probability", use_container_width=True)

    with st.expander("Preprocessing details"):
        detail_col_a, detail_col_b = st.columns(2)
        with detail_col_a:
            st.image(preprocess_result.grayscale_image, caption="Enhanced grayscale", use_container_width=True)
        with detail_col_b:
            st.image(preprocess_result.foreground_crop, caption="Detected object crop", use_container_width=True)
        st.json(
            {
                "selected_mode": mode_label,
                "background_option": background_label,
                "inverted": preprocess_result.was_inverted,
                "otsu_used": preprocess_result.used_otsu,
                "cleanup_threshold": cleanup_threshold,
            }
        )


if __name__ == "__main__":
    render_prediction()
