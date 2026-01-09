"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the data directory path."""

    return get_project_root() / "data"


def get_images_dir() -> Path:
    """Return the images directory path."""

    return get_data_dir() / "images"
