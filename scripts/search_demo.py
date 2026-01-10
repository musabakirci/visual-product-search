"""CLI demo for similarity search."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import select

from src.db.models import Product
from src.db.session import get_session
from src.embedding.extractor import extract_embedding
from src.search.retrieval import log_similarity_results, search_similar_products
from src.services.logging_service import configure_logging

logger = configure_logging()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a similarity search demo.")
    parser.add_argument("image_path", help="Path to a query image.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    return parser.parse_args()


def main() -> None:
    """Run the similarity search demo."""

    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # 1️⃣ Query embedding
    embedding = extract_embedding(str(image_path))

    # 2️⃣ Run search + fetch product metadata INSIDE session
    with get_session() as session:
        results = search_similar_products(session, embedding, top_k=args.top_k)
        log_similarity_results(session, str(image_path), results)

        # Build plain-dict product map (NO ORM leakage)
        product_map: dict[int, dict] = {}
        if results:
            product_ids = [item["product_id"] for item in results]
            stmt = select(Product).where(Product.id.in_(product_ids))
            for product in session.execute(stmt).scalars().all():
                product_map[product.id] = {
                    "id": product.id,
                    "name": product.name,
                }

    if not results:
        logger.info("No results found. Ensure embeddings are built.")
        return

    # 3️⃣ CLI output (safe: dict only)
    for item in results:
        product = product_map.get(item["product_id"])
        name = product["name"] if product else "Unknown"
        score = item["similarity_score"]
        print(f"{item['rank']}. product_id={item['product_id']} name={name} score={score:.4f}")


if __name__ == "__main__":
    main()
