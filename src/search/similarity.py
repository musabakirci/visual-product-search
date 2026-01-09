"""Similarity utilities."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute cosine similarity between two vectors."""

    a_vec = a.reshape(-1).astype(np.float32, copy=False)
    b_vec = b.reshape(-1).astype(np.float32, copy=False)
    denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)) + epsilon
    return float(np.dot(a_vec, b_vec) / denom)
