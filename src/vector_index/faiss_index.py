"""FAISS index helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build a FAISS IndexFlatIP from embeddings."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings provided.")

    vectors = np.ascontiguousarray(embeddings.astype(np.float32))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    logger.info("Built FAISS index with %s vectors", vectors.shape[0])
    return index


def save_faiss_index(index: Any, path: str) -> None:
    """Persist a FAISS index to disk."""

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(dest))
    logger.info("Saved FAISS index to %s", dest)


def load_faiss_index(path: str) -> Any:
    """Load a FAISS index from disk."""

    index_path = Path(path)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    index = faiss.read_index(str(index_path))
    logger.info("Loaded FAISS index from %s", index_path)
    return index
