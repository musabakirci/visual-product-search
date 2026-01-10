"""Offline clustering script for product embeddings."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.clustering.assignment import assign_clusters
from src.clustering.kmeans import run_kmeans
from src.config import resolve_embedding_scope
from src.db.models import ImageEmbedding
from src.db.session import get_session
from src.services.logging_service import configure_logging

logger = configure_logging()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build clusters for product embeddings.")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters.")
    return parser.parse_args()


def _decode_embedding(record: ImageEmbedding) -> np.ndarray:
    """Decode embedding bytes into a numpy vector."""

    vector = np.frombuffer(record.embedding, dtype=np.float32)
    if record.embedding_dim and vector.size != record.embedding_dim:
        logger.warning(
            "Embedding dim mismatch for product_id=%s expected=%s got=%s",
            record.product_id,
            record.embedding_dim,
            vector.size,
        )
        if vector.size < record.embedding_dim:
            return np.array([], dtype=np.float32)
        vector = vector[: record.embedding_dim]
    return vector


def load_embeddings(
    session: Session,
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> Tuple[list[int], np.ndarray]:
    """Load all embeddings and return product IDs with a 2D array."""

    resolved_version, resolved_type = resolve_embedding_scope(
        embedding_version,
        embedding_type,
    )
    stmt = select(ImageEmbedding).where(
        ImageEmbedding.embedding_version == resolved_version,
        ImageEmbedding.embedding_type == resolved_type,
    )
    records = session.execute(stmt).scalars().all()
    product_ids: list[int] = []
    vectors: list[np.ndarray] = []
    for record in records:
        vector = _decode_embedding(record)
        if vector.size == 0:
            continue
        product_ids.append(record.product_id)
        vectors.append(vector)

    if not vectors:
        return product_ids, np.empty((0, 0), dtype=np.float32)

    return product_ids, np.vstack(vectors)


def build_clusters(n_clusters: int) -> None:
    """Compute clusters and store cluster assignments."""

    with get_session() as session:
        product_ids, embeddings = load_embeddings(session)
        if embeddings.size == 0:
            logger.warning("No embeddings found. Build embeddings first.")
            return

        labels = run_kmeans(embeddings, n_clusters=n_clusters)
        updated = assign_clusters(session, product_ids, labels)

    logger.info("Embeddings clustered: %s", len(product_ids))
    logger.info("Clusters requested: %s", n_clusters)
    logger.info("Assignments completed: %s", updated)


if __name__ == "__main__":
    args = parse_args()
    build_clusters(args.clusters)
