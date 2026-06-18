import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from feature_extraction import (
    extract_combined_features,
    export_extracted_features_csv,
    build_rf_feature_vector,
    get_rf_prob_column_names,
    get_visual_feature_column_names,
)
from backend.app.config import BASE_DATASET_DIR

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DATASET = os.path.join(_REPO_ROOT, "Mango_data", "train")
IMAGE_DIR = _LOCAL_DATASET if os.path.isdir(_LOCAL_DATASET) else BASE_DATASET_DIR
METADATA_CSV = "mango_metadata.csv"
FEATURES_CSV = "extracted_features.csv"
CNN_MODEL_PATH = "mango_ripeness_mobilenetv2.keras"
RF_MODEL_PATH = "random_forest_harvest.joblib"
RF_COLUMNS_PATH = "rf_feature_columns.joblib"


def _cnn_predict_probs(cnn_model, image_path):
    """RGB uint8 input — matches predict.py and the Keras preprocess_input layer."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_batch = np.expand_dims(img_resized, axis=0)
    return cnn_model.predict(img_batch, verbose=0)[0]


def build_regression_dataset(cnn_model, force_reextract=False):
    """
    Step 1: Visual features -> extracted_features.csv
    Step 2: Merge with mango_metadata.csv on filename
    Step 3: Append CNN probabilities and environmental columns for RF training
    """
    print("--- Step 1: Extracting visual features (HSV averages + histograms + LBP) ---")
    if force_reextract or not os.path.exists(FEATURES_CSV):
        export_extracted_features_csv(METADATA_CSV, IMAGE_DIR, FEATURES_CSV)
    else:
        print(f"Using existing {FEATURES_CSV} (pass force_reextract=True to rebuild)")

    print("--- Step 2: Merging visual features with environmental metadata ---")
    df_features = pd.read_csv(FEATURES_CSV)
    df_meta = pd.read_csv(METADATA_CSV)

    df_merged = df_meta.merge(df_features, on="filename", how="inner")
    if df_merged.empty:
        raise RuntimeError(
            "Merged dataset is empty. Ensure filenames in mango_metadata.csv "
            "match extracted_features.csv and images exist under IMAGE_DIR."
        )

    print(f"Merged {len(df_merged)} rows (metadata x visual features)")

    visual_cols = [c for c in df_features.columns if c != "filename"]
    prob_cols = get_rf_prob_column_names()
    env_cols = ["temperature", "humidity"]
    target_col = "days_to_harvest"

    master_rows = []
    feature_columns = None

    print("--- Step 3: Adding MobileNetV2 probabilities per image ---")
    for index, row in df_merged.iterrows():
        img_path = os.path.join(IMAGE_DIR, str(row["filename"]))
        cnn_probs = _cnn_predict_probs(cnn_model, img_path)
        if cnn_probs is None:
            print(f"Skipping {img_path}: could not load image for CNN.")
            continue

        visual_vector = row[visual_cols].to_numpy(dtype=np.float32)
        combined = build_rf_feature_vector(
            cnn_probs,
            visual_vector,
            row["temperature"],
            row["humidity"],
        )
        master_rows.append(np.hstack([combined, row[target_col]]))

        if feature_columns is None:
            feature_columns = prob_cols + visual_cols + env_cols

        if (len(master_rows)) % 50 == 0:
            print(f"Processed {len(master_rows)} / {len(df_merged)} images...")

    col_names = feature_columns + [target_col]
    df_master = pd.DataFrame(master_rows, columns=col_names)
    print(f"Training matrix shape: {df_master.shape}\n")
    return df_master, feature_columns


def train_random_forest(df_master, feature_columns):
    """Train Random Forest on visual + CNN + environmental features; y = days_to_harvest."""
    print("--- Step 4: Training Random Forest Regressor ---")

    target_col = "days_to_harvest"
    X = df_master[feature_columns]
    y = df_master[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"Mean Squared Error (MSE):  {mse:.2f}")
    print(f"R-squared (R2 Score):      {r2:.2f}")
    print(f"Interpretation: On average, the model's harvest prediction is off by {mae:.2f} days.")

    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(feature_columns, RF_COLUMNS_PATH)
    print(f"\nRandom Forest saved to {RF_MODEL_PATH}")
    print(f"Feature column order saved to {RF_COLUMNS_PATH}")


def main(force_reextract=False):
    if not os.path.exists(METADATA_CSV):
        print(f"Error: Could not find {METADATA_CSV}. Run create_mock_metadata.py first.")
        return

    print(f"Loading CNN model from {CNN_MODEL_PATH}...")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

    df_master, feature_columns = build_regression_dataset(
        cnn_model, force_reextract=force_reextract
    )
    train_random_forest(df_master, feature_columns)


if __name__ == "__main__":
    main(force_reextract=True)
