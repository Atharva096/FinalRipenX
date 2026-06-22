import cv2
import numpy as np

WRONG_INPUT_MESSAGE = "Wrong input entered"

# Minimum share of pixels that look like mango skin (green / yellow / orange).
MANGO_MIN_PIXEL_RATIO = 0.10


def is_mango_image(image_rgb: np.ndarray) -> tuple[bool, str]:
    """
    Heuristic check that the image likely contains a mango.
    Uses HSV color ranges aligned with the project's ripeness hue bands.
    """
    if image_rgb is None or image_rgb.size == 0:
        return False, WRONG_INPUT_MESSAGE

    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        return False, WRONG_INPUT_MESSAGE

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    ripe_or_partial = (hue >= 8) & (hue <= 40)
    unripe_green = (hue >= 40) & (hue <= 85)
    mango_colored = (ripe_or_partial | unripe_green) & (saturation >= 25) & (value >= 35)

    mango_ratio = float(np.count_nonzero(mango_colored)) / float(mango_colored.size)
    if mango_ratio < MANGO_MIN_PIXEL_RATIO:
        return False, WRONG_INPUT_MESSAGE

    return True, "OK"


def ensure_mango_image(image_rgb: np.ndarray) -> None:
    """Raise ValueError when the image does not appear to contain a mango."""
    is_mango, message = is_mango_image(image_rgb)
    if not is_mango:
        raise ValueError(message)
