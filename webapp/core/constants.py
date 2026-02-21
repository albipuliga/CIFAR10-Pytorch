"""Application-wide constants."""

from __future__ import annotations

APP_NAME = "CIFAR-10 Portfolio API"
APP_VERSION = "0.1.0"

CIFAR10_CLASSES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MODEL_CHECKPOINT_FILENAMES = {
    "baseline": "best_baseline.pth",
    "cnnv2": "best_cnnv2.pth",
}

DEFAULT_MODEL_ID = "cnnv2"
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}

INPUT_IMAGE_SIZE = (32, 32)
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
