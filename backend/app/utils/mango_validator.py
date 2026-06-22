"""
Mango image validation using MobileNetV2 ImageNet embeddings.

Compares uploaded images against a precomputed centroid built from the
training dataset. This reliably rejects posters, people, cars, etc. that
color-only heuristics miss.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WRONG_INPUT_MESSAGE = "Wrong input entered. Please upload a valid mango image."
VALIDATOR_VERSION = "2.0-embedding"
MIN_EMBEDDING_SIMILARITY = 0.55  # Lowered slightly for better recall

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CENTROID_PATH = _DATA_DIR / "mango_centroid.npy"

_embed_model = None
_mango_centroid: np.ndarray | None = None
_validator_ready = False


def _load_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    _embed_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    return _embed_model


def init_mango_validator() -> None:
    """Load centroid and embedding model once at startup."""
    global _mango_centroid, _validator_ready

    logger.info("Initializing mango validator...")
    logger.info("Looking for centroid at: %s", _CENTROID_PATH)
    
    if not _CENTROID_PATH.is_file():
        logger.error("Mango centroid missing at %s", _CENTROID_PATH)
        logger.error("Please run scripts/build_mango_centroid.py to generate it")
        _validator_ready = False
        return

    try:
        _mango_centroid = np.load(_CENTROID_PATH).astype(np.float32)
        norm = np.linalg.norm(_mango_centroid)
        if norm > 0:
            _mango_centroid = _mango_centroid / norm
        
        logger.info("Centroid loaded successfully (shape: %s)", _mango_centroid.shape)

        _load_embed_model()
        _validator_ready = True
        logger.info("✅ Mango validator %s ready (min_similarity=%.2f)", 
                   VALIDATOR_VERSION, MIN_EMBEDDING_SIMILARITY)
    except Exception as exc:
        logger.error("❌ Failed to initialize mango validator: %s", exc)
        import traceback
        logger.error(traceback.format_exc())
        _validator_ready = False


def validator_status() -> dict:
    return {
        "version": VALIDATOR_VERSION,
        "ready": _validator_ready,
        "centroid_loaded": _mango_centroid is not None,
        "min_similarity": MIN_EMBEDDING_SIMILARITY,
        "centroid_path": str(_CENTROID_PATH),
    }


def _image_to_rgb_array(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError(WRONG_INPUT_MESSAGE)
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError(WRONG_INPUT_MESSAGE)
    return cv2.resize(image_rgb, (224, 224))


def _compute_embedding(image_rgb: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model = _load_embed_model()
    arr = _image_to_rgb_array(image_rgb)
    batch = preprocess_input(arr.astype(np.float32))
    batch = np.expand_dims(batch, axis=0)
    vector = model.predict(batch, verbose=0)[0].astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError(WRONG_INPUT_MESSAGE)
    return vector / norm


def embedding_similarity(image_rgb: np.ndarray) -> float:
    if _mango_centroid is None:
        raise RuntimeError("Mango validator is not initialized")
    embedding = _compute_embedding(image_rgb)
    return float(np.dot(embedding, _mango_centroid))


def is_mango_image(image_rgb: np.ndarray) -> tuple[bool, str, float]:
    """
    Check if image is a mango.
    Returns: (is_mango, message, similarity_score)
    """
    if not _validator_ready or _mango_centroid is None:
        logger.warning("⚠️  Embedding validator unavailable; using strict color fallback")
        is_valid, msg = _color_fallback(image_rgb)
        return is_valid, msg, 0.0

    try:
        similarity = embedding_similarity(image_rgb)
        logger.info("🔍 Embedding similarity score: %.3f (threshold: %.2f)", 
                   similarity, MIN_EMBEDDING_SIMILARITY)
    except ValueError:
        return False, WRONG_INPUT_MESSAGE, 0.0
    except Exception as exc:
        logger.error("Embedding validation failed: %s", exc)
        is_valid, msg = _color_fallback(image_rgb)
        return is_valid, msg, 0.0

    if similarity < MIN_EMBEDDING_SIMILARITY:
        logger.info("❌ Rejected non-mango image (similarity=%.3f)", similarity)
        return False, f"Not a mango image (similarity: {similarity:.2f} < {MIN_EMBEDDING_SIMILARITY})", similarity

    logger.info("✅ Valid mango image (similarity=%.3f)", similarity)
    return True, "OK", similarity


def _color_fallback(image_rgb: np.ndarray) -> tuple[bool, str]:
    """Strict HSV fallback when embedding model cannot run."""
    try:
        arr = _image_to_rgb_array(image_rgb)
    except ValueError:
        return False, WRONG_INPUT_MESSAGE

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    colorful = (saturation >= 35) & (value >= 45)
    if int(np.count_nonzero(colorful)) < 80:
        return False, WRONG_INPUT_MESSAGE

    mango_skin = colorful & (
        ((hue >= 8) & (hue <= 42)) | ((hue >= 42) & (hue <= 78))
    )
    blue = colorful & (hue >= 95) & (hue <= 130)
    red = colorful & ((hue <= 8) | (hue >= 172))

    total = float(hue.size)
    mango_ratio = float(np.count_nonzero(mango_skin)) / total
    blue_ratio = float(np.count_nonzero(blue)) / total
    red_ratio = float(np.count_nonzero(red)) / total

    if mango_ratio < 0.18 or blue_ratio > 0.04 or red_ratio > 0.10:
        return False, WRONG_INPUT_MESSAGE

    return True, "OK"


def ensure_mango_image(image_rgb: np.ndarray) -> tuple[bool, str, float]:
    """
    Validate image and return detailed result.
    Returns: (is_valid, message, similarity_score)
    """
    return is_mango_image(image_rgb)


def ensure_mango_prediction(_image_rgb: np.ndarray, _predicted_class: str) -> None:
    """Kept for API compatibility; embedding gate runs before prediction."""
    return None