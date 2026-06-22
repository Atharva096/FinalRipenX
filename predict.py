import os
import cv2
import numpy as np
import tensorflow as tf
import joblib

from feature_extraction import extract_combined_features
from export_recommendation import (
    DEFAULT_CULTIVAR,
    get_mandatory_regulatory_compliance,
    recommend_export_destination,
)
from harvest_inference import (
    estimate_harvest_days,
    format_harvest_message,
    load_rf_feature_columns,
)

TEST_IMAGE_PATH = r"C:\Users\Izana\Documents\Final-BE-Project\913.jpeg"
ENVIRONMENT_TEMP = 35
ENVIRONMENT_HUMIDITY = 50

CNN_MODEL_PATH = "mango_ripeness_mobilenetv2.keras"
RF_MODEL_PATH = "random_forest_harvest.joblib"
RF_COLUMNS_PATH = "rf_feature_columns.joblib"

CLASS_NAMES = {
    0: "Partially Ripe",
    1: "Ripe",
    2: "Unripe",
}


def print_regulatory_compliance_block(destination: str) -> None:
    """Print mandatory phytosanitary actions when USA, Japan, or EU is in the route."""
    actions = get_mandatory_regulatory_compliance(destination)
    if not actions:
        return
    print(" Mandatory Regulatory Compliance")
    for action in actions:
        print(f" {action}")


def predict_mango_system(image_path, temp, humidity, cultivar: str = DEFAULT_CULTIVAR):
    print("=" * 50)
    print(" MANGO RIPENESS & HARVEST PREDICTOR")
    print("=" * 50)

    print("\n[1/4] Loading trained models...")
    if not os.path.exists(CNN_MODEL_PATH) or not os.path.exists(RF_MODEL_PATH):
        print("Error: Could not find model files. Train Phases 3 & 4 first.")
        return

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    feature_columns = load_rf_feature_columns(RF_COLUMNS_PATH)

    print(f"[2/4] Reading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image. Check TEST_IMAGE_PATH.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_batch = np.expand_dims(img_resized, axis=0)

    print("[3/4] Running MobileNetV2 classification...")
    cnn_preds = cnn_model.predict(img_batch, verbose=0)[0]
    predicted_class_idx = int(np.argmax(cnn_preds))
    predicted_class_name = CLASS_NAMES.get(predicted_class_idx, "Unknown")
    confidence = float(cnn_preds[predicted_class_idx] * 100)
    print(
        f"      -> Result: This mango is {predicted_class_name} ({confidence:.1f}% confidence)."
    )

    print("[4/4] Extracting texture/color features and estimating harvest time...")
    visual_features = extract_combined_features(image_path)
    if visual_features is None:
        print("Error: Failed to extract features.")
        return

    final_days, raw_days = estimate_harvest_days(
        rf_model,
        cnn_preds,
        visual_features,
        temp,
        humidity,
        predicted_class_name,
        feature_columns=feature_columns,
    )

    export_dest, export_logistics = recommend_export_destination(final_days, cultivar)

    print("\n" + "=" * 50)
    print(" FINAL PREDICTION REPORT")
    print("=" * 50)
    print(format_harvest_message(predicted_class_name, temp, humidity, final_days, raw_days))
    print(f" Recommended Export Destination: {export_dest}")
    print(f" Logistics Action Required: {export_logistics}")
    print_regulatory_compliance_block(export_dest)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    predict_mango_system(TEST_IMAGE_PATH, ENVIRONMENT_TEMP, ENVIRONMENT_HUMIDITY)
