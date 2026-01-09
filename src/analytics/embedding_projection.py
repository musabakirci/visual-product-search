"""Embedding projection utilities."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

try:
    import umap  # type: ignore

    _UMAP_AVAILABLE = True
except Exception:
    _UMAP_AVAILABLE = False


def _default_perplexity(n_samples: int) -> int:
    """Select a safe perplexity for t-SNE."""

    if n_samples < 2:
        raise ValueError("At least two samples are required for projection.")
    return max(1, min(30, n_samples - 1))


def project_embeddings(
    embeddings: np.ndarray,
    method: str = "tsne",
    random_state: int = 42,
) -> np.ndarray:
    """Project embeddings into 2D using t-SNE or UMAP."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[0] < 2:
        raise ValueError("At least two embeddings are required.")

    method_key = method.lower()
    logger.info("Projecting embeddings using %s", method_key)
    data = embeddings.astype(np.float32, copy=False)

    if method_key == "tsne":
        perplexity = _default_perplexity(data.shape[0])
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=1000,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        projection = model.fit_transform(data)
    elif method_key == "umap":
        if not _UMAP_AVAILABLE:
            raise ImportError("UMAP is not available. Install umap-learn to use it.")
        model = umap.UMAP(n_components=2, random_state=random_state)
        projection = model.fit_transform(data)
    else:
        raise ValueError(f"Unsupported projection method: {method}")

    return projection.astype(np.float32, copy=False)
