import cv2
import numpy as np
from feature_extraction import extract_hsv_features, extract_lbp_features, extract_combined_features

# --- 1. Choose a real image from your dataset ---
import os
from backend.app.config import BASE_DATASET_DIR

# --- 1. Choose a real image from your dataset ---
# This automatically creates: D:\RipenX\mango_dataset\mango_data\train\ripe\327(4).jpg
IMAGE_PATH = BASE_DATASET_DIR
def main():
    print(f"Loading image from: {IMAGE_PATH}\n")
    image = cv2.imread(IMAGE_PATH)
    
    if image is None:
        print("Error: Could not load image. Please double-check the IMAGE_PATH.")
        return

    # Resize just like we do in the main pipeline
    image = cv2.resize(image, (224, 224))

    # --- 2. Extract Features ---
    hsv_features = extract_hsv_features(image)
    lbp_features = extract_lbp_features(image)
    combined_features = extract_combined_features(IMAGE_PATH)

    # --- 3. Print a clean summary ---
    print("="*40)
    print(" FEATURE EXTRACTION SUMMARY")
    print("="*40)
    print(f"Total Combined Features: {len(combined_features)} values\n")

    print(f" HSV (Color) Features [Length: {len(hsv_features)}]")
    print(f"First 10 values: {np.round(hsv_features[:10], 5)}\n")

    print(f" LBP (Texture) Features [Length: {len(lbp_features)}]")
    print(f"First 10 values: {np.round(lbp_features[:10], 5)}\n")

    

if __name__ == "__main__":
    main()