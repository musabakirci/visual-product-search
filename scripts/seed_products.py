"""Seed sample products into the database."""

from __future__ import annotations

from sqlalchemy import or_, select

from src.db.models import Base, Product
from src.db.session import engine, get_session
from src.services.logging_service import configure_logging

logger = configure_logging()

SAMPLE_PRODUCTS = [
     # Shoes
    {
        "name": "Sports Shoe 1",
        "category": "shoes",
        "image_path": "data/images/foto1.jpg",
    },
    {
        "name": "Sports Shoe 2",
        "category": "shoes",
        "image_path": "data/images/foto2.jpg",
    },
    {
        "name": "Sports Shoe 3",
        "category": "shoes",
        "image_path": "data/images/foto3.jpg",
    },

    # Jerseys
    {
        "name": "Football Jersey 1",
        "category": "jersey",
        "image_path": "data/images/foto4.jpg",
    },
    {
        "name": "Football Jersey 2",
        "category": "jersey",
        "image_path": "data/images/foto5.jpg",
    },

    # Accessories (beanie & scarf)
    {
        "name": "Winter Beanie 1",
        "category": "accessories",
        "image_path": "data/images/foto6.jpg",
    },
    {
        "name": "Winter Pants 1",
        "category": "pants",
        "image_path": "data/images/foto7.jpg",
    },
    {
        "name": "Winter Pants 2",
        "category": "pants",
        "image_path": "data/images/foto8.jpg",
    },
    {
        "name": "Winter Beanie 2",
        "category": "accessories",
        "image_path": "data/images/foto9.jpg",
    },
    {
        "name": "Winter Scarf 1",
        "category": "accessories",
        "image_path": "data/images/foto10.jpg",
    },
    {
        "name": "Winter Scarf 2",
        "category": "accessories",
        "image_path": "data/images/foto11.jpg",
    },
]


def seed_products() -> int:
    """Insert sample products in an idempotent way."""

    Base.metadata.create_all(bind=engine)
    inserted = 0
    with get_session() as session:
        for item in SAMPLE_PRODUCTS:
            stmt = select(Product).where(
                or_(
                    Product.name == item["name"],
                    Product.image_path == item["image_path"],
                )
            )
            existing = session.execute(stmt).scalars().first()
            if existing:
                continue
            session.add(Product(**item))
            inserted += 1
    return inserted


if __name__ == "__main__":
    count = seed_products()
    logger.info("Seeded %s products", count)
