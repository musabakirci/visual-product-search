"""Application configuration loading."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class Settings(BaseModel):
    database_url: str = "sqlite:///./app.db"
    app_name: str = "Visual Product Search"
    embedding_version: str = "v1"
    embedding_type: str = "resnet50"


@lru_cache()
def get_settings() -> Settings:
    load_dotenv()
    data = {
        "database_url": os.getenv("DATABASE_URL", Settings().database_url),
        "app_name": os.getenv("APP_NAME", Settings().app_name),
        "embedding_version": os.getenv("EMBEDDING_VERSION", Settings().embedding_version),
        "embedding_type": os.getenv("EMBEDDING_TYPE", Settings().embedding_type),
    }
    try:
        return Settings(**data)
    except ValidationError as exc:
        logging.getLogger(__name__).error("Invalid configuration: %s", exc)
        raise


def resolve_embedding_scope(
    embedding_version: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> tuple[str, str]:
    settings = get_settings()
    return (
        embedding_version or settings.embedding_version,
        embedding_type or settings.embedding_type,
    )
