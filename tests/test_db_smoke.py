"""Basic database smoke test."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Product


def test_db_smoke() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, future=True)

    with SessionLocal() as session:
        product = Product(
            name="Test Product",
            category="Test Category",
            image_path="data/images/test.jpg",
        )
        session.add(product)
        session.commit()
        session.refresh(product)

        fetched = session.get(Product, product.id)
        assert fetched is not None
        assert fetched.name == "Test Product"
