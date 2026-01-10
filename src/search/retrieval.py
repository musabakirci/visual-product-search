"""Top-N retrieval for similar products."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import resolve_embedding_scope
from src.db.crud import log_similarity_event
from src.db.models import ImageEmbedding
from src.search.similarity import cosine_similarity
from src.utils.paths import get_data_dir
from src.vector_index.faiss_index import load_faiss_index
from src.vector_index.faiss_search import faiss_search

logger = logging.getLogger(__name__)


def _decode_embedding(record: ImageEmbedding) -> np.ndarray:
    """Decode an embedding BLOB into a numpy array."""

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


def _iter_embeddings(
    session: Session,
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> Iterable[ImageEmbedding]:
    """Yield embedding records from the database."""

    resolved_version, resolved_type = resolve_embedding_scope(
        embedding_version,
        embedding_type,
    )
    stmt = select(ImageEmbedding).where(
        ImageEmbedding.embedding_version == resolved_version,
        ImageEmbedding.embedding_type == resolved_type,
    )
    return session.execute(stmt).scalars().all()


@lru_cache(maxsize=1)
def _load_faiss_assets() -> Tuple[object, np.ndarray]:
    """Load FAISS index and product IDs from disk."""

    faiss_dir = get_data_dir() / "faiss"
    index_path = faiss_dir / "index.bin"
    ids_path = faiss_dir / "product_ids.npy"
    index = load_faiss_index(str(index_path))
    product_ids = np.load(ids_path)
    return index, product_ids


def _attach_cluster_ids(
    session: Session,
    results: list[dict],
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> list[dict]:
    """Attach cluster IDs to result items when available."""

    if not results:
        return results
    product_ids = [item["product_id"] for item in results]
    resolved_version, resolved_type = resolve_embedding_scope(
        embedding_version,
        embedding_type,
    )
    stmt = select(ImageEmbedding.product_id, ImageEmbedding.cluster_id).where(
        ImageEmbedding.product_id.in_(product_ids),
        ImageEmbedding.embedding_version == resolved_version,
        ImageEmbedding.embedding_type == resolved_type,
    )
    cluster_map = {
        product_id: cluster_id for product_id, cluster_id in session.execute(stmt).all()
    }
    for item in results:
        item["cluster_id"] = cluster_map.get(item["product_id"])
    return results

def _prepare_entries(
    session: Session,
    query_size: int,
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> list[dict]:
    """Prepare embedding entries with decoded vectors."""

    entries: list[dict] = []
    for record in _iter_embeddings(
        session,
        embedding_version=embedding_version,
        embedding_type=embedding_type,
    ):
        vector = _decode_embedding(record)
        if vector.size == 0:
            continue
        if vector.size != query_size:
            logger.warning(
                "Skipping product_id=%s due to dimension mismatch query=%s embed=%s",
                record.product_id,
                query_size,
                vector.size,
            )
            continue
        entries.append(
            {
                "product_id": record.product_id,
                "cluster_id": record.cluster_id,
                "vector": vector,
            }
        )
    return entries


def _compute_centroids(entries: list[dict]) -> dict[int, np.ndarray]:
    """Compute normalized centroids for each cluster."""

    clusters: dict[int, list[np.ndarray]] = {}
    for entry in entries:
        cluster_id = entry["cluster_id"]
        if cluster_id is None:
            continue
        clusters.setdefault(int(cluster_id), []).append(entry["vector"])

    centroids: dict[int, np.ndarray] = {}
    for cluster_id, vectors in clusters.items():
        matrix = np.vstack(vectors)
        centroid = matrix.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        centroids[cluster_id] = centroid / norm
    return centroids


def _select_cluster(query: np.ndarray, centroids: dict[int, np.ndarray]) -> Optional[int]:
    """Select the best cluster for a query embedding."""

    if not centroids:
        return None

    best_cluster: Optional[int] = None
    best_score = float("-inf")
    for cluster_id, centroid in centroids.items():
        score = cosine_similarity(query, centroid)
        if score > best_score:
            best_score = score
            best_cluster = cluster_id
    return best_cluster


def _assign_ranks(results: list[dict]) -> list[dict]:
    """Assign ranks to results in-place."""

    for idx, item in enumerate(results, start=1):
        item["rank"] = idx
    return results


def search_similar_products(
    session: Session,
    query_embedding: np.ndarray,
    top_k: int = 5,
    same_cluster_first: bool = True,
    use_faiss: bool = False,
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> list[dict]:
    """Search for top-k similar products by cosine similarity."""

    if top_k <= 0:
        return []

    query = query_embedding.reshape(-1).astype(np.float32, copy=False)
    if use_faiss and not same_cluster_first:
        try:
            index, product_ids = _load_faiss_assets()
            results = faiss_search(query, top_k, index, product_ids)
            return _attach_cluster_ids(
                session,
                results,
                embedding_version=embedding_version,
                embedding_type=embedding_type,
            )
        except Exception as exc:
            logger.warning("FAISS search unavailable, falling back to brute-force: %s", exc)

    if use_faiss and same_cluster_first:
        logger.info(
            "Cluster-prioritized search uses brute-force fallback when FAISS is enabled."
        )

    entries = _prepare_entries(
        session,
        query.size,
        embedding_version=embedding_version,
        embedding_type=embedding_type,
    )
    if not entries:
        return []

    results: list[dict] = []
    for entry in entries:
        score = cosine_similarity(query, entry["vector"])
        results.append(
            {
                "product_id": entry["product_id"],
                "cluster_id": entry["cluster_id"],
                "similarity_score": float(score),
                "rank": 0,
            }
        )

    results.sort(key=lambda item: item["similarity_score"], reverse=True)
    if not same_cluster_first:
        return _assign_ranks(results[:top_k])

    centroids = _compute_centroids(entries)
    selected_cluster = _select_cluster(query, centroids)
    if selected_cluster is None:
        return _assign_ranks(results[:top_k])

    prioritized: list[dict] = []
    seen: set[int] = set()
    for item in results:
        if item["cluster_id"] == selected_cluster:
            prioritized.append(item)
            seen.add(item["product_id"])
            if len(prioritized) == top_k:
                return _assign_ranks(prioritized)

    for item in results:
        if item["product_id"] in seen:
            continue
        prioritized.append(item)
        if len(prioritized) == top_k:
            break

    return _assign_ranks(prioritized)


def log_similarity_results(
    session: Session,
    query_image_path: str,
    results: list[dict],
) -> None:
    """Persist similarity results into the log table."""

    for item in results:
        log_similarity_event(
            session=session,
            query_image_path=query_image_path,
            matched_product_id=item["product_id"],
            similarity_score=item["similarity_score"],
            rank=item["rank"],
        )
    session.commit()
