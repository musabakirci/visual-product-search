"""Cluster assignment persistence."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.config import resolve_embedding_scope
from src.db.models import ImageEmbedding

logger = logging.getLogger(__name__)


def assign_clusters(
    session: Session,
    product_ids: list[int],
    cluster_labels: np.ndarray,
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> int:
    """Assign cluster IDs to embeddings for the given products."""

    if len(product_ids) != len(cluster_labels):
        raise ValueError("Product IDs and cluster labels must have the same length.")

    resolved_version, resolved_type = resolve_embedding_scope(
        embedding_version,
        embedding_type,
    )
    updated = 0
    for product_id, cluster_id in zip(product_ids, cluster_labels.tolist()):
        stmt = (
            update(ImageEmbedding)
            .where(
                ImageEmbedding.product_id == product_id,
                ImageEmbedding.embedding_version == resolved_version,
                ImageEmbedding.embedding_type == resolved_type,
            )
            .values(cluster_id=int(cluster_id))
        )
        result = session.execute(stmt)
        if result.rowcount:
            updated += result.rowcount

    session.commit()
    logger.info("Assigned clusters for %s embeddings", updated)
    return updated
