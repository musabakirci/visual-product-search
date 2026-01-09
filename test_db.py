from src.db.session import get_session
from src.db.models import Product
from sqlalchemy import select

with get_session() as session:
    products = session.execute(select(Product)).scalars().all()
    print("Products:", len(products))
