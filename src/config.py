"""Application configuration loading."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class Settings(BaseModel):
    """Typed application settings."""

    database_url: str = "sqlite:///./app.db"
    app_name: str = "Visual Product Search"


@lru_cache()
def get_settings() -> Settings:
    """Load settings from environment variables and .env."""

    load_dotenv()
    data = {
        "database_url": os.getenv("DATABASE_URL", Settings().database_url),
        "app_name": os.getenv("APP_NAME", Settings().app_name),
    }
    try:
        return Settings(**data)
    except ValidationError as exc:
        logging.getLogger(__name__).error("Invalid configuration: %s", exc)
        raise
