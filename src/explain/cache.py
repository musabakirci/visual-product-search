"""Disk cache helpers for explanations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image

from src.utils.paths import get_data_dir


def _cache_dir() -> Path:
    """Return the explanation cache directory."""

    path = get_data_dir() / "explanations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_key(image_path: str, model_name: str) -> str:
    """Generate a stable cache key for an image and model."""

    path = Path(image_path).resolve()
    value = f"{path}::{model_name}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_cached_explanation(key: str) -> Optional[Path]:
    """Return a cached explanation path if it exists."""

    path = _cache_dir() / f"{key}.png"
    return path if path.exists() else None


def save_explanation_image(key: str, image: Image.Image) -> Path:
    """Persist an explanation image to disk."""

    path = _cache_dir() / f"{key}.png"
    image.save(path)
    return path
