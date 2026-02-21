"""Image validation and preprocessing utilities."""

from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from webapp.core.constants import ALLOWED_IMAGE_MIME_TYPES, INPUT_IMAGE_SIZE


class UnsupportedMediaTypeError(ValueError):
    """Raised when an uploaded file has an unsupported MIME type."""


class UploadTooLargeError(ValueError):
    """Raised when an uploaded payload exceeds max size."""


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a valid image."""


def validate_upload(content_type: str | None, raw_bytes: bytes, max_upload_bytes: int) -> None:
    if content_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise UnsupportedMediaTypeError(
            "Unsupported file type. Upload a PNG or JPEG image."
        )
    if len(raw_bytes) > max_upload_bytes:
        raise UploadTooLargeError(
            f"Upload too large. Maximum size is {max_upload_bytes // (1024 * 1024)} MB."
        )


def image_bytes_to_tensor(
    image_bytes: bytes,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidImageError("Uploaded file is not a valid image.") from exc

    transform = transforms.Compose(
        [
            transforms.Resize(INPUT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform(image).unsqueeze(0)
