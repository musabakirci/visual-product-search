"""Offline FAISS index builder."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import resolve_embedding_scope
from src.db.models import ImageEmbedding
from src.db.session import get_session
from src.services.logging_service import configure_logging
from src.utils.paths import get_data_dir
from src.vector_index.faiss_index import build_faiss_index, save_faiss_index

logger = configure_logging()


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and product IDs in matching order."""

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
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.vstack(vectors), np.array(product_ids, dtype=np.int32)


def build_index() -> None:
    """Build and persist a FAISS index."""

    with get_session() as session:
        embeddings, product_ids = load_embeddings(session)

    if embeddings.size == 0:
        logger.warning("No embeddings found. Build embeddings before indexing.")
        return

    index = build_faiss_index(embeddings)
    faiss_dir = get_data_dir() / "faiss"
    index_path = faiss_dir / "index.bin"
    ids_path = faiss_dir / "product_ids.npy"
    save_faiss_index(index, str(index_path))
    np.save(ids_path, product_ids)

    size_bytes = Path(index_path).stat().st_size
    logger.info("Embeddings indexed: %s", embeddings.shape[0])
    logger.info("Index path: %s", index_path)
    logger.info("Index size (bytes): %s", size_bytes)


if __name__ == "__main__":
    build_index()
