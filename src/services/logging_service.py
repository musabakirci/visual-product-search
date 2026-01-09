"""Logging configuration."""

from __future__ import annotations

import logging
from typing import Union


def configure_logging(level: Union[int, str] = "INFO") -> logging.Logger:
    """Configure application logging and return the root logger."""

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    return logging.getLogger("app")
