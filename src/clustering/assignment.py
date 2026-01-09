"""Cluster assignment persistence."""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.db.models import ImageEmbedding

logger = logging.getLogger(__name__)


def assign_clusters(
    session: Session,
    product_ids: list[int],
    cluster_labels: np.ndarray,
) -> int:
    """Assign cluster IDs to embeddings for the given products."""

    if len(product_ids) != len(cluster_labels):
        raise ValueError("Product IDs and cluster labels must have the same length.")

    updated = 0
    for product_id, cluster_id in zip(product_ids, cluster_labels.tolist()):
        stmt = (
            update(ImageEmbedding)
            .where(ImageEmbedding.product_id == product_id)
            .values(cluster_id=int(cluster_id))
        )
        result = session.execute(stmt)
        if result.rowcount:
            updated += result.rowcount

    session.commit()
    logger.info("Assigned clusters for %s embeddings", updated)
    return updated
