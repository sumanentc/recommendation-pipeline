#!/usr/bin/env python
"""Plain-Python entry point: runs the whole pipeline (or selected stages) in dependency order.

Examples
--------
    python run_pipeline.py                                  # everything
    python run_pipeline.py --stages train evaluate report   # only some stages (inputs must already exist)
    python run_pipeline.py --set ingestion.source=synthetic # override any config value
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.config import load_config  # noqa: E402
from src.pipeline import STAGE_NAMES, run_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to a YAML config (default: configs/config.yaml)")
    ap.add_argument("--stages", nargs="+", choices=STAGE_NAMES, help="run only these stages")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="override config values, e.g. --set ingestion.source=synthetic evaluation.primary_k=5")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_all(cfg, args.stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
