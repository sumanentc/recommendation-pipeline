"""Configuration helpers shared by every pipeline stage."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``a.b.c=value`` style overrides (value parsed as YAML)."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = apply_overrides(copy.deepcopy(cfg), overrides)
    cfg["_root"] = str(ROOT)
    cfg["_config_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    """Short, stable fingerprint of the configuration (ignores private keys)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


def data_dir(cfg: dict, key: str) -> Path:
    """Return (and create) one of the directories listed under ``paths``."""
    p = Path(cfg["_root"]) / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
