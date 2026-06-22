import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
_LOCAL_DATASET = BASE_DIR / "Mango_data" / "train"
_DOWNLOADS_DATASET = Path(r"C:\Users\Izana\Downloads\Final-BE-Project\Mango_data\train")
BASE_DATASET_DIR = str(
    _LOCAL_DATASET if _LOCAL_DATASET.is_dir() else _DOWNLOADS_DATASET
)
# Same Keras model as predict.py (prefer repo root).
_LOCAL_MODEL = BASE_DIR / "mango_ripeness_mobilenetv2.keras"
_DOWNLOADS_MODEL = Path(
    r"C:\Users\Izana\Downloads\Final-BE-Project\mango_ripeness_mobilenetv2.keras"
)
MODEL_PATH = _LOCAL_MODEL if _LOCAL_MODEL.is_file() else _DOWNLOADS_MODEL
# Class index -> label order must match the exported model's id2label mapping.
# mango_ripeness_model/config.json:
#   0 -> "partially_ripe"
#   1 -> "ripe"
#   2 -> "unripe"
# Human-readable class names (matching `predict.py`).
RIPENESS_CLASSES = ["Partially Ripe", "Ripe", "Unripe"]
RF_MODEL_PATH = BASE_DIR / "random_forest_harvest.joblib"
RF_COLUMNS_PATH = BASE_DIR / "rf_feature_columns.joblib"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
