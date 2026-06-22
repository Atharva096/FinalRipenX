"""Build mango_centroid.npy from the local Mango_data training set."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "Mango_data" / "train"
OUTPUT = Path(__file__).resolve().parents[1] / "app" / "data" / "mango_centroid.npy"


def embed(model, image_path: Path) -> np.ndarray:
    image = cv2.resize(np.array(Image.open(image_path).convert("RGB")), (224, 224))
    batch = preprocess_input(image.astype(np.float32))
    vector = model.predict(np.expand_dims(batch, axis=0), verbose=0)[0]
    return vector / (np.linalg.norm(vector) + 1e-8)


def main() -> None:
    if not DATASET.is_dir():
        raise SystemExit(f"Dataset not found: {DATASET}")

    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",
    )

    vectors = []
    for folder in ("ripe", "unripe", "partially_ripe"):
        class_dir = DATASET / folder
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir())[:40]:
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            vectors.append(embed(model, path))

    if not vectors:
        raise SystemExit("No training images found to build centroid.")

    centroid = np.mean(vectors, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, centroid.astype(np.float32))
    print(f"Saved {OUTPUT} from {len(vectors)} mango images.")


if __name__ == "__main__":
    main()
