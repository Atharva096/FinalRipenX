"""Shared harvest-time regression + biological sanity checks (predict.py & API)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from backend.feature_extraction import build_rf_feature_vector


def apply_harvest_sanity_check(status_name: str, raw_days: float, temperature: float) -> int:
    """
    Post-process RF days-to-harvest using ripeness class and ambient temperature.
    Ripe fruit at high heat should not show many days until ready.
    """
    days = float(raw_days)

    if status_name == "Ripe":
        if temperature >= 35:
            days = min(days, 1.0)
        elif temperature >= 30:
            days = min(days, 1.5)
        else:
            days = min(days, 2.0)
        return int(max(0, round(days)))

    if status_name == "Partially Ripe":
        if temperature >= 35:
            days = min(days, max(1.0, days * 0.6))
        return int(max(0, round(days)))

    return int(max(0, round(days)))


def build_rf_input(
    cnn_probs,
    visual_features,
    temperature: float,
    humidity: float,
    feature_columns: Optional[list] = None,
):
    """Feature vector (and optional named DataFrame) for random_forest_harvest.joblib."""
    feature_vector = build_rf_feature_vector(
        cnn_probs, visual_features, temperature, humidity
    )
    if feature_columns is not None:
        if len(feature_columns) != len(feature_vector):
            raise ValueError(
                f"Feature count mismatch: model expects {len(feature_columns)}, "
                f"got {len(feature_vector)}. Retrain with train_rf_harvest.py."
            )
        return pd.DataFrame([feature_vector], columns=feature_columns)
    return feature_vector.reshape(1, -1)


def estimate_harvest_days(
    rf_model,
    cnn_probs,
    visual_features,
    temperature: float,
    humidity: float,
    ripeness_class: str,
    feature_columns: Optional[list] = None,
) -> Tuple[int, float]:
    """
    Run RF regression then sanity check.
    Returns (final_days, raw_rf_days).
    """
    X = build_rf_input(
        cnn_probs,
        visual_features,
        temperature,
        humidity,
        feature_columns=feature_columns,
    )
    raw_days = float(rf_model.predict(X)[0])
    final_days = apply_harvest_sanity_check(ripeness_class, raw_days, temperature)
    return final_days, raw_days


def load_rf_feature_columns(rf_columns_path: os.PathLike | str) -> Optional[list]:
    path = os.fspath(rf_columns_path)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def format_harvest_message(
    ripeness_class: str,
    temperature: float,
    humidity: float,
    harvest_days: int,
    raw_days: Optional[float] = None,
) -> str:
    """Human-readable report matching predict.py wording."""
    if harvest_days <= 0:
        harvest_line = "Harvest Estimate:   Ready to harvest TODAY!"
    elif harvest_days == 1:
        harvest_line = "Harvest Estimate:   Ready within about 1 day."
    else:
        harvest_line = f"Harvest Estimate:   {harvest_days} days until ready."

    lines = [
        f"Status:             {ripeness_class}",
        f"Current Temp:       {temperature:g}\u00b0C",
        f"Current Humidity:   {humidity:g}%",
        harvest_line,
    ]
    if (
        raw_days is not None
        and ripeness_class == "Ripe"
        and int(round(raw_days)) != harvest_days
    ):
        lines.append(
            f"(RF raw estimate: {raw_days:.1f} days — adjusted for ripe fruit at {temperature:g}\u00b0C)"
        )
    return "\n".join(lines)
