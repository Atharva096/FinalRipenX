from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_DATASET_DIR = BASE_DIR / "Mango_data" / "train"

MODEL_PATH = BASE_DIR / "app" / "mango_ripeness_mobilenetv2.keras"

RF_MODEL_PATH = BASE_DIR / "random_forest_harvest.joblib"
RF_COLUMNS_PATH = BASE_DIR / "rf_feature_columns.joblib"

RIPENESS_CLASSES = [
    "Partially Ripe",
    "Ripe",
    "Unripe"
]

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024

print(f"MODEL_PATH: {MODEL_PATH}")
print(f"MODEL EXISTS: {MODEL_PATH.exists()}")