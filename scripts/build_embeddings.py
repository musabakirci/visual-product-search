"""Batch embedding generation script."""

from __future__ import annotations

from pathlib import Path

from src.db.crud import list_products
from src.db.session import get_session
from src.embedding.extractor import extract_embedding
from src.embedding.storage import save_embedding
from src.services.logging_service import configure_logging

logger = configure_logging()


def resolve_image_path(image_path: str) -> Path:
    """Resolve image path to an absolute path."""

    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def build_embeddings() -> None:
    """Generate embeddings for all products with valid images."""

    created = 0
    skipped_missing = 0
    skipped_existing = 0

    with get_session() as session:
        products = list_products(session)
        total = len(products)
        for product in products:
            image_path = resolve_image_path(product.image_path)
            if not image_path.exists():
                skipped_missing += 1
                logger.warning(
                    "Missing image for product_id=%s path=%s",
                    product.id,
                    image_path,
                )
                continue

            try:
                embedding = extract_embedding(str(image_path))
            except Exception as exc:
                skipped_missing += 1
                logger.warning(
                    "Failed to process image for product_id=%s path=%s error=%s",
                    product.id,
                    image_path,
                    exc,
                )
                continue

            created_now = save_embedding(
                session=session,
                product_id=product.id,
                embedding=embedding,
                model_name="resnet50",
            )
            if created_now:
                created += 1
            else:
                skipped_existing += 1

    logger.info("Total products: %s", total)
    logger.info("Embeddings created: %s", created)
    logger.info("Skipped missing/unreadable: %s", skipped_missing)
    logger.info("Skipped existing: %s", skipped_existing)


if __name__ == "__main__":
    build_embeddings()
