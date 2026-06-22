import os
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from export_recommendation import (
    DEFAULT_CULTIVAR,
    format_regulatory_compliance_block,
    get_mandatory_regulatory_compliance,
    recommend_export_destination,
)
from feature_extraction import extract_combined_features
from harvest_inference import (
    estimate_harvest_days,
    format_harvest_message,
    load_rf_feature_columns,
)

CNN_MODEL_PATH = "mango_ripeness_mobilenetv2.keras"
RF_MODEL_PATH = "random_forest_harvest.joblib"
RF_COLUMNS_PATH = "rf_feature_columns.joblib"

CLASS_NAMES = {
    0: "Partially Ripe",
    1: "Ripe",
    2: "Unripe",
}

CULTIVAR_OPTIONS = ["Alphonso", "Kesar", "Totapuri", "Other"]


@st.cache_resource
def load_models():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if not Path(CNN_MODEL_PATH).is_file() or not Path(RF_MODEL_PATH).is_file():
        return None, None, None
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    feature_columns = load_rf_feature_columns(RF_COLUMNS_PATH)
    return cnn_model, rf_model, feature_columns


def _save_upload_to_temp(mango_image) -> str:
    suffix = Path(mango_image.name or "capture.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(mango_image.getvalue())
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise


def analyze_mango(mango_image, temperature: float, humidity: float, cultivar: str):
    cnn_model, rf_model, feature_columns = load_models()
    if cnn_model is None:
        st.error("Model files not found. Train MobileNet and Random Forest first.")
        return

    temp_path = _save_upload_to_temp(mango_image)
    try:
        image = cv2.imread(temp_path)
        if image is None:
            st.error("Could not read the image. Try another photo or format.")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(image_rgb, (224, 224))
        img_batch = np.expand_dims(img_resized, axis=0)

        cnn_preds = cnn_model.predict(img_batch, verbose=0)[0]
        predicted_class_idx = int(np.argmax(cnn_preds))
        predicted_class_name = CLASS_NAMES.get(predicted_class_idx, "Unknown")
        confidence = float(cnn_preds[predicted_class_idx])

        visual_features = extract_combined_features(temp_path)
        if visual_features is None:
            st.error("Failed to extract image features.")
            return

        final_days, raw_days = estimate_harvest_days(
            rf_model,
            cnn_preds,
            visual_features,
            temperature,
            humidity,
            predicted_class_name,
            feature_columns=feature_columns,
        )

        export_dest, export_logistics = recommend_export_destination(final_days, cultivar)
        regulatory_actions = get_mandatory_regulatory_compliance(export_dest)

        st.subheader("Prediction report")
        st.metric("Ripeness", predicted_class_name)
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.metric("Days until ready", final_days)

        st.text(format_harvest_message(
            predicted_class_name, temperature, humidity, final_days, raw_days
        ))

        st.markdown(f"**Recommended Export Destination:** {export_dest}")
        st.markdown(f"**Logistics Action Required:** {export_logistics}")

        if regulatory_actions:
            st.markdown("**Mandatory Regulatory Compliance**")
            for action in regulatory_actions:
                st.markdown(f"- {action}")

    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    st.set_page_config(page_title="RipenX Mango Analyzer", page_icon="🥭", layout="wide")
    st.title("Mango Ripeness & Harvest Predictor")
    st.caption("Upload a file or take a live photo, then run analysis.")

    col1, col2 = st.columns(2)

    with col1:
        file_input = st.file_uploader(
            "Upload a mango photo",
            type=["jpg", "jpeg", "png", "bmp"],
        )

    with col2:
        camera_input = st.camera_input("Or click a live photo")

    mango_image = file_input or camera_input

    st.divider()

    env_col1, env_col2, env_col3 = st.columns(3)
    with env_col1:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=35.0)
    with env_col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    with env_col3:
        cultivar = st.selectbox("Cultivar", CULTIVAR_OPTIONS, index=0)

    if mango_image is not None:
        preview = Image.open(mango_image).convert("RGB")
        st.image(preview, caption="Selected mango image", use_container_width=True)
        mango_image.seek(0)

    analyze = st.button("Analyze Mango", type="primary", disabled=mango_image is None)

    if analyze and mango_image is not None:
        with st.spinner("Running ripeness and harvest analysis..."):
            analyze_mango(mango_image, temperature, humidity, cultivar)
    elif mango_image is None:
        st.info("Choose a file upload or capture a live photo to begin.")


if __name__ == "__main__":
    main()
