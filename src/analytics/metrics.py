"""Analytics metrics for search behavior."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from src.db.models import SimilarityLog

logger = logging.getLogger(__name__)


def total_searches(session: Session) -> int:
    """Return the total number of similarity search log entries."""

    count = session.execute(select(func.count(SimilarityLog.id))).scalar_one()
    return int(count)


def average_similarity(session: Session) -> float:
    """Return the average similarity score."""

    value = session.execute(select(func.avg(SimilarityLog.similarity_score))).scalar()
    return float(value) if value is not None else 0.0


def top_matched_products(session: Session, limit: int = 5) -> list[dict[str, Any]]:
    """Return the most frequently matched products."""

    stmt = (
        select(
            SimilarityLog.matched_product_id.label("product_id"),
            func.count(SimilarityLog.id).label("match_count"),
        )
        .where(SimilarityLog.matched_product_id.is_not(None))
        .group_by(SimilarityLog.matched_product_id)
        .order_by(desc("match_count"))
        .limit(limit)
    )
    rows = session.execute(stmt).all()
    results = [{"product_id": row.product_id, "match_count": row.match_count} for row in rows]
    logger.info("Top matched products retrieved: %s", len(results))
    return results


def searches_over_time(session: Session, freq: str = "hour") -> list[dict[str, Any]]:
    """Return search counts aggregated by hour or day."""

    if freq == "hour":
        bucket = func.strftime("%Y-%m-%d %H:00:00", SimilarityLog.created_at)
    elif freq == "day":
        bucket = func.strftime("%Y-%m-%d", SimilarityLog.created_at)
    else:
        raise ValueError("Unsupported frequency. Use 'hour' or 'day'.")

    stmt = (
        select(bucket.label("bucket"), func.count(SimilarityLog.id).label("count"))
        .group_by("bucket")
        .order_by("bucket")
    )
    rows = session.execute(stmt).all()
    return [{"bucket": row.bucket, "count": row.count} for row in rows]
