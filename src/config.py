"""Shared project settings and path helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_SEED = 42
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def configure_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """Configure consistent console logging for command-line modules."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
