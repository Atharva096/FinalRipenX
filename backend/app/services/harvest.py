"""Harvest estimation helpers for the FastAPI app."""

from app.config import RF_COLUMNS_PATH
from backend.harvest_inference import (
    estimate_harvest_days,
    format_harvest_message,
    load_rf_feature_columns,
)

_rf_feature_columns = None


def get_rf_feature_columns():
    global _rf_feature_columns
    if _rf_feature_columns is None:
        _rf_feature_columns = load_rf_feature_columns(RF_COLUMNS_PATH)
    return _rf_feature_columns


def predict_harvest(
    rf_model,
    cnn_probs,
    visual_features,
    temperature: float,
    humidity: float,
    ripeness_class: str,
):
    final_days, raw_days = estimate_harvest_days(
        rf_model,
        cnn_probs,
        visual_features,
        temperature,
        humidity,
        ripeness_class,
        feature_columns=get_rf_feature_columns(),
    )
    message = format_harvest_message(
        ripeness_class, temperature, humidity, final_days, raw_days=raw_days
    )
    return final_days, raw_days, message
