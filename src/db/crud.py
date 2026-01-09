"""Basic CRUD helpers."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import ImageEmbedding, Product, SimilarityLog


def create_product(
    session: Session,
    name: str,
    category: Optional[str],
    image_path: str,
) -> Product:
    """Create a new product."""

    product = Product(name=name, category=category, image_path=image_path)
    session.add(product)
    session.flush()
    session.refresh(product)
    return product


def list_products(session: Session, limit: Optional[int] = None) -> list[dict]:
    stmt = select(Product).order_by(Product.id)
    if limit is not None:
        stmt = stmt.limit(limit)

    products = session.execute(stmt).scalars().all()
    return [serialize_product(p) for p in products]



def get_product(session: Session, product_id: int) -> Optional[Product]:
    """Get a product by ID."""

    return session.get(Product, product_id)


def create_embedding(
    session: Session,
    product_id: int,
    model_name: str,
    embedding_dim: int,
    embedding: bytes,
    cluster_id: Optional[int] = None,
) -> ImageEmbedding:
    """Create an embedding entry (placeholder)."""

    existing = get_embedding_by_product(session, product_id)
    if existing is not None:
        raise ValueError(f"Embedding already exists for product_id={product_id}")

    record = ImageEmbedding(
        product_id=product_id,
        model_name=model_name,
        embedding_dim=embedding_dim,
        embedding=embedding,
        cluster_id=cluster_id,
    )
    session.add(record)
    session.flush()
    session.refresh(record)
    return record


def get_embedding_by_product(
    session: Session,
    product_id: int,
) -> Optional[ImageEmbedding]:
    """Get an embedding by product ID."""

    stmt = select(ImageEmbedding).where(ImageEmbedding.product_id == product_id)
    return session.execute(stmt).scalars().first()


def log_similarity_event(
    session: Session,
    query_image_path: str,
    matched_product_id: Optional[int],
    similarity_score: Optional[float],
    rank: Optional[int],
) -> SimilarityLog:
    """Log a similarity search event."""

    log_entry = SimilarityLog(
        query_image_path=query_image_path,
        matched_product_id=matched_product_id,
        similarity_score=similarity_score,
        rank=rank,
    )
    session.add(log_entry)
    session.flush()
    session.refresh(log_entry)
    return log_entry

def serialize_product(product: Product) -> dict:
    return {
        "id": product.id,
        "name": product.name,
        "category": product.category,
        "image_path": product.image_path,
    }
