"""MLflow helpers (shared by training and evaluation)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep the console output clean

import mlflow  # noqa: E402


def setup_mlflow(cfg: dict) -> str:
    """Point MLflow at a local SQLite DB inside the repo and make sure the experiment exists."""
    root = Path(cfg["_root"])
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(root / uri[len("sqlite:///"):])
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = root / "mlruns"
        artifacts.mkdir(exist_ok=True)
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)
    return uri
