"""Embedding extraction logic."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.embedding.model import get_device, get_embedding_model
from src.embedding.preprocess import preprocess_image

logger = logging.getLogger(__name__)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized vector."""

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize zero-length vector.")
    return vector / norm


def extract_embedding(image_path: str) -> np.ndarray:
    """Extract a normalized embedding for a single image."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = get_embedding_model()
    device = get_device()
    tensor = preprocess_image(str(path)).to(device)

    with torch.no_grad():
        features = model(tensor)

    embedding = features.squeeze(0).cpu().numpy().astype(np.float32)
    if embedding.ndim != 1:
        embedding = embedding.reshape(-1)
    embedding = _l2_normalize(embedding)
    return embedding
