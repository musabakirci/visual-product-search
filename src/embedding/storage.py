"""Persistence helpers for embeddings."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from src.config import resolve_embedding_scope
from src.db.crud import get_embedding_by_product
from src.db.models import ImageEmbedding

logger = logging.getLogger(__name__)


def save_embedding(
    session: Session,
    product_id: int,
    embedding: np.ndarray,
    model_name: str,
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> bool:
    """Save an embedding if one does not already exist."""

    resolved_version, resolved_type = resolve_embedding_scope(
        embedding_version,
        embedding_type,
    )
    existing = get_embedding_by_product(
        session,
        product_id,
        embedding_version=resolved_version,
        embedding_type=resolved_type,
    )
    if existing is not None:
        logger.info("Embedding already exists for product_id=%s", product_id)
        return False

    vector = embedding.astype(np.float32).reshape(-1)
    record = ImageEmbedding(
        product_id=product_id,
        model_name=model_name,
        embedding_dim=int(vector.size),
        embedding=vector.tobytes(),
        embedding_version=resolved_version,
        embedding_type=resolved_type,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return True
