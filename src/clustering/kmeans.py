"""K-Means clustering utilities."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def run_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run K-Means clustering and return cluster labels."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings provided.")

    cluster_count = min(n_clusters, embeddings.shape[0])
    if cluster_count != n_clusters:
        logger.warning(
            "Requested clusters (%s) exceeds sample count (%s). Using %s clusters.",
            n_clusters,
            embeddings.shape[0],
            cluster_count,
        )

    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels.astype(int)
