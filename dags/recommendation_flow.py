"""Prefect orchestration of the recommendation pipeline.

DAG (each arrow = "needs the result of"):

    ingest -> validate -> prepare -> transform -> features -> train -> evaluate -> report

* every stage is a Prefect *task* that receives the result of its upstream task, so Prefect knows the
  dependencies and draws them in the UI
* if a task fails (e.g. the validation quality gate) all downstream tasks are NOT run
* `ingest` is retried twice (network hiccups), the others are deterministic and are not retried

Run it
------
    python dags/recommendation_flow.py                 # run once (ephemeral local Prefect, no server needed)
    python dags/recommendation_flow.py --serve         # keep running and trigger daily at 02:00 (cron)

To watch runs in the Prefect UI (for the screenshot) start the server in a second terminal first:
    prefect server start
    export PREFECT_API_URL=http://127.0.0.1:4200/api   # in the terminal that runs the flow
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Prefect's anonymous telemetry can print a harmless SQLAlchemy traceback while the temporary local
# server starts up; switching it off keeps the console clean.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, task, get_run_logger  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.pipeline import write_manifest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.reporting import make_reports  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.validation import validate  # noqa: E402


def _run_stage(name, fn, config_path, overrides):
    logger = get_run_logger()
    logger.info("starting stage '%s'", name)
    summary = fn(load_config(config_path, overrides))
    logger.info("finished stage '%s' in %ss", name, summary.get("seconds", "?"))
    return summary


# `upstream` is not used inside the function - it only creates the dependency edge in the DAG.
@task(name="1-ingest", retries=2, retry_delay_seconds=5)
def ingest_task(config_path, overrides, upstream=None):
    return _run_stage("ingest", ingest.run, config_path, overrides)


@task(name="2-validate")
def validate_task(config_path, overrides, upstream=None):
    return _run_stage("validate", validate.run, config_path, overrides)


@task(name="3-prepare")
def prepare_task(config_path, overrides, upstream=None):
    return _run_stage("prepare", prepare.run, config_path, overrides)


@task(name="4-transform")
def transform_task(config_path, overrides, upstream=None):
    return _run_stage("transform", transform.run, config_path, overrides)


@task(name="5-features")
def features_task(config_path, overrides, upstream=None):
    return _run_stage("features", build_features.run, config_path, overrides)


@task(name="6-train")
def train_task(config_path, overrides, upstream=None):
    return _run_stage("train", train.run, config_path, overrides)


@task(name="7-evaluate")
def evaluate_task(config_path, overrides, upstream=None):
    return _run_stage("evaluate", evaluate.run, config_path, overrides)


@task(name="8-report")
def report_task(config_path, overrides, upstream=None):
    return _run_stage("report", make_reports.run, config_path, overrides)


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: Optional[str] = None, overrides: Optional[List[str]] = None) -> dict:
    s1 = ingest_task(config_path, overrides)
    s2 = validate_task(config_path, overrides, upstream=s1)
    s3 = prepare_task(config_path, overrides, upstream=s2)
    s4 = transform_task(config_path, overrides, upstream=s3)
    s5 = features_task(config_path, overrides, upstream=s4)
    s6 = train_task(config_path, overrides, upstream=s5)
    s7 = evaluate_task(config_path, overrides, upstream=s6)
    s8 = report_task(config_path, overrides, upstream=s7)
    summaries = {"ingest": s1, "validate": s2, "prepare": s3, "transform": s4,
                 "features": s5, "train": s6, "evaluate": s7, "report": s8}
    write_manifest(load_config(config_path, overrides), summaries, "success", "prefect")
    return summaries


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="+", default=None, dest="overrides", metavar="KEY=VALUE")
    ap.add_argument("--serve", action="store_true", help="serve the flow with a daily 02:00 cron schedule")
    args = ap.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="recommendation-pipeline-daily", cron="0 2 * * *",
                                      parameters={"config_path": args.config, "overrides": args.overrides})
    else:
        recommendation_pipeline(args.config, args.overrides)
