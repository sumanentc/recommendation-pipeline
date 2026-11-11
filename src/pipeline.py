"""Stage registry + plain-Python runner (used by run_pipeline.py and by the tests).

The Prefect flow in dags/recommendation_flow.py wraps the very same stage functions, so both
orchestrators execute identical code.
"""
from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
import time
import traceback

from src.common.config import ROOT, config_hash, data_dir
from src.common.io import utc_now_iso, write_json
from src.common.log import get_logger
from src.evaluation import evaluate
from src.features import build_features
from src.ingestion import ingest
from src.preparation import prepare
from src.reporting import make_reports
from src.training import train
from src.transformation import transform
from src.validation import validate

log = get_logger("pipeline")

# (name, function) in dependency order: every stage consumes the files written by the previous one
STAGES = [
    ("ingest", ingest.run),
    ("validate", validate.run),
    ("prepare", prepare.run),
    ("transform", transform.run),
    ("features", build_features.run),
    ("train", train.run),
    ("evaluate", evaluate.run),
    ("report", make_reports.run),
]
STAGE_NAMES = [n for n, _ in STAGES]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "not-a-git-repo"


def _versions() -> dict:
    pkgs = ["pandas", "numpy", "scipy", "scikit-learn", "pyarrow", "mlflow", "prefect", "PyYAML"]
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "not installed"
    return out


def write_manifest(cfg: dict, summaries: dict, status: str, orchestrator: str) -> None:
    write_json(data_dir(cfg, "reports") / "run_manifest.json", {
        "finished_at": utc_now_iso(), "status": status, "orchestrator": orchestrator,
        "config_hash": config_hash(cfg), "git_commit": _git_commit(),
        "python": sys.version.split()[0], "platform": platform.platform(), "packages": _versions(),
        "stages": summaries,
    })


def run_all(cfg: dict, stages: list[str] | None = None) -> dict:
    """Run the stages sequentially. A failing stage stops the run (downstream stages are skipped)."""
    wanted = stages or STAGE_NAMES
    unknown = set(wanted) - set(STAGE_NAMES)
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}; choose from {STAGE_NAMES}")
    summaries: dict = {}
    t0 = time.time()
    status = "success"
    for name, fn in STAGES:
        if name not in wanted:
            continue
        log.info("=" * 20 + f" stage: {name} " + "=" * 20)
        t = time.time()
        try:
            summaries[name] = fn(cfg)
        except Exception as exc:
            status = f"failed at '{name}': {type(exc).__name__}: {exc}"
            summaries[name] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            log.error("Stage '%s' FAILED -> %s", name, exc)
            write_manifest(cfg, summaries, status, "run_pipeline.py")
            raise
        log.info("stage '%s' finished in %.1fs", name, time.time() - t)
    write_manifest(cfg, summaries, status, "run_pipeline.py")
    log.info("PIPELINE %s in %.1fs", status.upper(), time.time() - t0)
    return summaries
