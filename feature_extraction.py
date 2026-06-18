import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
import pandas as pd

# --- Configuration ---
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

HSV_BINS = (8, 8, 8)
HSV_HIST_LEN = HSV_BINS[0] * HSV_BINS[1] * HSV_BINS[2]
LBP_HIST_LEN = LBP_N_POINTS + 2  # uniform LBP -> P + 2 distinct labels

# Column names for HSV channel averages (Hue maps to degradation / ripening time)
HSV_AVERAGE_COLUMNS = ["mean_hue", "mean_saturation", "mean_value"]


def extract_hsv_channel_averages(image):
    """
    Mean Hue, Saturation, and Value for the image.
    Hue is especially useful as a linear proxy for color-driven ripening progress.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = cv2.mean(hsv_image)[:3]
    return np.array([h_mean, s_mean, v_mean], dtype=np.float32)


def extract_hsv_features(image):
    """3D HSV histogram features (normalized)."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv_image],
        channels=[0, 1, 2],
        mask=None,
        histSize=HSV_BINS,
        ranges=[0, 180, 0, 256, 0, 256],
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_lbp_features(image):
    """Local Binary Pattern texture histogram (fixed bin count for uniform LBP)."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    n_bins = LBP_HIST_LEN
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist


def _load_and_resize_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.resize(image, (224, 224))


def extract_combined_features(image_path):
    """
    Visual feature vector: HSV channel averages + HSV histogram + LBP histogram.
    Order is fixed so training and inference stay aligned.
    """
    image = _load_and_resize_image(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        return None

    hsv_averages = extract_hsv_channel_averages(image)
    hsv_features = extract_hsv_features(image)
    lbp_features = extract_lbp_features(image)
    return np.hstack([hsv_averages, hsv_features, lbp_features])


def get_visual_feature_column_names():
    """Column names for the visual portion of extracted_features.csv."""
    cols = list(HSV_AVERAGE_COLUMNS)
    cols += [f"hsv_hist_{i}" for i in range(HSV_HIST_LEN)]
    cols += [f"lbp_{i}" for i in range(LBP_HIST_LEN)]
    return cols


VISUAL_FEATURE_LEN = len(HSV_AVERAGE_COLUMNS) + HSV_HIST_LEN + LBP_HIST_LEN


def build_rf_feature_vector(cnn_probs, visual_features, temperature, humidity):
    """
    Single 1D vector for random_forest_harvest.joblib.
    Order: [CNN probs (3), visual features, temperature, humidity]
    """
    return np.hstack(
        [
            np.asarray(cnn_probs, dtype=np.float32).ravel(),
            np.asarray(visual_features, dtype=np.float32).ravel(),
            np.float32(temperature),
            np.float32(humidity),
        ]
    )


def get_rf_prob_column_names():
    """Matches MobileNet folder order: partially_ripe=0, ripe=1, unripe=2."""
    return ["prob_partially_ripe", "prob_ripe", "prob_unripe"]


def get_rf_feature_column_names():
    """Full ordered feature names for the Random Forest regressor."""
    return get_rf_prob_column_names() + get_visual_feature_column_names() + [
        "temperature",
        "humidity",
    ]


def export_extracted_features_csv(
    metadata_csv,
    image_dir,
    output_csv="extracted_features.csv",
):
    """
    Batch-extract visual features for every filename in mango_metadata.csv.
    Writes mean Hue/Saturation/Value plus HSV/LBP histograms.
    """
    df_meta = pd.read_csv(metadata_csv)
    if "filename" not in df_meta.columns:
        raise ValueError(f"{metadata_csv} must contain a 'filename' column")

    rows = []
    visual_cols = get_visual_feature_column_names()

    for index, row in df_meta.iterrows():
        img_path = os.path.join(image_dir, str(row["filename"]))
        features = extract_combined_features(img_path)
        if features is None:
            print(f"Skipping {img_path}: could not extract features.")
            continue

        if len(features) != VISUAL_FEATURE_LEN:
            raise ValueError(
                f"Unexpected feature length {len(features)} for {img_path} "
                f"(expected {VISUAL_FEATURE_LEN})"
            )

        record = {"filename": row["filename"]}
        for col, val in zip(visual_cols, features):
            record[col] = val
        rows.append(record)

        if (index + 1) % 50 == 0:
            print(f"Extracted features for {index + 1} / {len(df_meta)} images...")

    if not rows:
        raise RuntimeError("No features were extracted. Check image paths and metadata.")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} rows to {output_csv} ({df_out.shape[1] - 1} visual features per image)")
    return df_out


if __name__ == "__main__":
    from app.config import BASE_DATASET_DIR

    METADATA_CSV = "mango_metadata.csv"
    OUTPUT_CSV = "extracted_features.csv"

    dummy_img_path = "test_mango_dummy.jpg"
    cv2.imwrite(dummy_img_path, np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    print("Testing feature extraction pipeline...")
    features = extract_combined_features(dummy_img_path)
    if features is not None:
        print(f"Success! Extracted a feature vector of length: {len(features)}")
        print(
            f"  HSV averages (H,S,V): {np.round(features[:3], 2)}"
        )

    if os.path.exists(dummy_img_path):
        os.remove(dummy_img_path)

    if os.path.exists(METADATA_CSV):
        print(f"\nBuilding {OUTPUT_CSV} from {METADATA_CSV}...")
        export_extracted_features_csv(METADATA_CSV, BASE_DATASET_DIR, OUTPUT_CSV)
