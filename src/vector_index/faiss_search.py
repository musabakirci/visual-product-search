"""FAISS search helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _normalize(vector: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize a vector for cosine similarity."""

    norm = np.linalg.norm(vector) + epsilon
    return vector / norm


def faiss_search(
    query_embedding: np.ndarray,
    top_k: int,
    index: Any,
    product_ids: np.ndarray,
) -> list[dict]:
    """Search a FAISS index and return ranked results."""

    if top_k <= 0:
        return []

    if product_ids.size == 0:
        return []

    query = query_embedding.reshape(1, -1).astype(np.float32, copy=False)
    query = np.ascontiguousarray(_normalize(query))
    index_size = getattr(index, "ntotal", int(product_ids.size))
    k = min(top_k, product_ids.size, int(index_size))
    scores, indices = index.search(query, k)
    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= product_ids.size:
            continue
        results.append(
            {
                "product_id": int(product_ids[idx]),
                "similarity_score": float(score),
                "rank": 0,
            }
        )
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    logger.info("FAISS search returned %s results", len(results))
    return results
