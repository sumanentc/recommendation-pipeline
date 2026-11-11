"""Tiny logging helper: console + logs/pipeline.log."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.common.config import ROOT

_FMT = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s", "%H:%M:%S")


def get_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(f"rec.{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_FMT)
    logger.addHandler(console)

    try:
        log_dir = Path(ROOT) / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(_FMT)
        logger.addHandler(file_handler)
    except OSError:  # read-only file system etc. - console logging is enough
        pass
    return logger
