import cv2
import numpy as np

WRONG_INPUT_MESSAGE = "Wrong input entered"

# OpenCV hue ranges (0-180) aligned with project ripeness bands.
RIPENESS_HUE_RANGES = {
    "Partially Ripe": (18, 42),
    "Ripe": (8, 28),
    "Unripe": (38, 72),
}

MANGO_MIN_COLORFUL_RATIO = 0.15
MAX_BLUE_RATIO = 0.05
MAX_RED_WITHOUT_MANGO = 0.08
MAX_HUE_STD_WITHOUT_MANGO = 32.0
MIN_RIPENESS_BAND_RATIO = 0.18


def _split_hsv(image_rgb: np.ndarray):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.split(hsv)


def _colorful_mask(saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    return (saturation >= 35) & (value >= 45)


def _mango_skin_mask(hue: np.ndarray, colorful: np.ndarray) -> np.ndarray:
    yellow_orange = (hue >= 8) & (hue <= 42)
    green_unripe = (hue >= 42) & (hue <= 78)
    return colorful & (yellow_orange | green_unripe)


def is_mango_image(image_rgb: np.ndarray) -> tuple[bool, str]:
    """
    Reject images that do not look like a mango photo.
    Filters out posters, logos, and other non-fruit scenes using HSV heuristics.
    """
    if image_rgb is None or image_rgb.size == 0:
        return False, WRONG_INPUT_MESSAGE

    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        return False, WRONG_INPUT_MESSAGE

    hue, saturation, value = _split_hsv(image_rgb)
    total_pixels = float(hue.size)
    colorful = _colorful_mask(saturation, value)
    colorful_count = int(np.count_nonzero(colorful))

    if colorful_count < 80:
        return False, WRONG_INPUT_MESSAGE

    mango_skin = _mango_skin_mask(hue, colorful)
    mango_ratio = float(np.count_nonzero(mango_skin)) / total_pixels
    mango_of_colorful = float(np.count_nonzero(mango_skin)) / float(colorful_count)

    blue = colorful & (hue >= 95) & (hue <= 130)
    red = colorful & ((hue <= 8) | (hue >= 172))
    blue_ratio = float(np.count_nonzero(blue)) / total_pixels
    red_ratio = float(np.count_nonzero(red)) / total_pixels
    hue_std = float(np.std(hue[colorful])) if colorful_count else 0.0

    if mango_ratio < MANGO_MIN_COLORFUL_RATIO:
        return False, WRONG_INPUT_MESSAGE

    if blue_ratio > MAX_BLUE_RATIO:
        return False, WRONG_INPUT_MESSAGE

    if red_ratio > MAX_RED_WITHOUT_MANGO and mango_of_colorful < 0.45:
        return False, WRONG_INPUT_MESSAGE

    if hue_std > MAX_HUE_STD_WITHOUT_MANGO and mango_of_colorful < 0.50:
        return False, WRONG_INPUT_MESSAGE

    return True, "OK"


def ripeness_matches_image_colors(image_rgb: np.ndarray, predicted_class: str) -> tuple[bool, str]:
    """
    Cross-check CNN output against dominant image colors.
    Non-mango images often get a high-confidence but wrong ripeness label.
    """
    hue_range = RIPENESS_HUE_RANGES.get(predicted_class)
    if hue_range is None:
        return False, WRONG_INPUT_MESSAGE

    hue, saturation, value = _split_hsv(image_rgb)
    colorful = _colorful_mask(saturation, value)
    colorful_count = int(np.count_nonzero(colorful))
    if colorful_count < 80:
        return False, WRONG_INPUT_MESSAGE

    low, high = hue_range
    in_band = colorful & (hue >= low - 10) & (hue <= high + 10)
    band_ratio = float(np.count_nonzero(in_band)) / float(colorful_count)
    mean_hue = float(np.mean(hue[colorful]))

    if band_ratio < MIN_RIPENESS_BAND_RATIO:
        return False, WRONG_INPUT_MESSAGE

    if not (low - 22 <= mean_hue <= high + 22):
        return False, WRONG_INPUT_MESSAGE

    return True, "OK"


def ensure_mango_image(image_rgb: np.ndarray) -> None:
    is_mango, message = is_mango_image(image_rgb)
    if not is_mango:
        raise ValueError(message)


def ensure_mango_prediction(image_rgb: np.ndarray, predicted_class: str) -> None:
    matches, message = ripeness_matches_image_colors(image_rgb, predicted_class)
    if not matches:
        raise ValueError(message)
