"""Offline embedding projection builder."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.analytics.embedding_projection import project_embeddings
from src.config import resolve_embedding_scope
from src.db.models import ImageEmbedding
from src.db.session import get_session
from src.services.logging_service import configure_logging
from src.utils.paths import get_data_dir

logger = configure_logging()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Build 2D projections for embeddings.")
    parser.add_argument("--method", default="tsne", help="Projection method: tsne or umap.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed.")
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings with product and cluster IDs."""

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
    cluster_ids: list[int] = []
    vectors: list[np.ndarray] = []
    for record in records:
        vector = _decode_embedding(record)
        if vector.size == 0:
            continue
        product_ids.append(record.product_id)
        cluster_ids.append(-1 if record.cluster_id is None else int(record.cluster_id))
        vectors.append(vector)

    if not vectors:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.vstack(vectors),
        np.array(product_ids, dtype=np.int32),
        np.array(cluster_ids, dtype=np.int32),
    )


def build_projection(method: str, random_state: int) -> None:
    """Build and persist 2D projections."""

    with get_session() as session:
        embeddings, product_ids, cluster_ids = load_embeddings(session)

    if embeddings.size == 0:
        logger.warning("No embeddings found. Build embeddings before projection.")
        return

    projection = project_embeddings(embeddings, method=method, random_state=random_state)
    output_dir = get_data_dir() / "projections"
    output_dir.mkdir(parents=True, exist_ok=True)
    projections_path = output_dir / "projections.npy"
    product_ids_path = output_dir / "product_ids.npy"
    cluster_ids_path = output_dir / "cluster_ids.npy"

    np.save(projections_path, projection)
    np.save(product_ids_path, product_ids)
    np.save(cluster_ids_path, cluster_ids)

    logger.info("Embeddings projected: %s", embeddings.shape[0])
    logger.info("Projection method: %s", method)
    logger.info("Projection path: %s", projections_path)


if __name__ == "__main__":
    args = parse_args()
    build_projection(args.method, args.random_state)
