"""Image loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGE_MEAN, std=_IMAGE_STD),
    ]
)


def preprocess_image(image_path: str) -> torch.Tensor:
    """Load an image from disk and apply standard transforms."""

    path = Path(image_path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = _TRANSFORM(image)
    return tensor.unsqueeze(0)
