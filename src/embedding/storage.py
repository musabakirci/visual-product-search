"""Persistence helpers for embeddings."""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy.orm import Session

from src.db.crud import get_embedding_by_product
from src.db.models import ImageEmbedding

logger = logging.getLogger(__name__)


def save_embedding(
    session: Session,
    product_id: int,
    embedding: np.ndarray,
    model_name: str,
) -> bool:
    """Save an embedding if one does not already exist."""

    existing = get_embedding_by_product(session, product_id)
    if existing is not None:
        logger.info("Embedding already exists for product_id=%s", product_id)
        return False

    vector = embedding.astype(np.float32).reshape(-1)
    record = ImageEmbedding(
        product_id=product_id,
        model_name=model_name,
        embedding_dim=int(vector.size),
        embedding=vector.tobytes(),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return True
