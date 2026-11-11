"""Stage 6 - Model training with experiment tracking (MLflow).

Trains: popularity baseline, SVD (matrix factorisation), hybrid (SVD + genre affinity).
For every model one MLflow run is created that stores
    * parameters  (hyper-parameters, seed, split strategy, data sizes, feature version)
    * metrics     (training time, explained variance, ...)  - test metrics are added by the evaluation stage
    * artifacts   (the pickled model, the config, the feature catalog)
The run ids are saved to models/training_manifest.json so that the evaluation stage can
log its metrics into the SAME runs.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import scipy.sparse as sp

from src.common.config import config_hash, data_dir
from src.common.io import read_json, utc_now_iso, write_json
from src.common.log import get_logger
from src.features.store import FeatureStore
from src.training.models import HybridRecommender, PopularityRecommender, SVDRecommender, TrainingData
from src.training.tracking import setup_mlflow

log = get_logger("training")


def load_training_data(cfg: dict) -> TrainingData:
    proc, feat = data_dir(cfg, "processed"), data_dir(cfg, "features")
    R = sp.load_npz(proc / "interaction_matrix.npz").tocsr()
    idx = np.load(proc / "matrix_index.npz")
    store = FeatureStore.load(feat, proc)
    return TrainingData(R=R, user_ids=idx["user_ids"], item_ids=idx["item_ids"], store=store,
                        rating_min=cfg["validation"]["rating_min"], rating_max=cfg["validation"]["rating_max"])


def build_models(cfg: dict) -> list:
    seed, m = cfg["project"]["seed"], cfg["models"]
    return [
        PopularityRecommender(),
        SVDRecommender(seed=seed, **m["svd"]),
        HybridRecommender(seed=seed, **m["hybrid"]),
    ]


def run(cfg: dict) -> dict:
    t0 = time.time()
    models_dir, proc, rep = data_dir(cfg, "models"), data_dir(cfg, "processed"), data_dir(cfg, "reports")
    tracking_uri = setup_mlflow(cfg)
    data = load_training_data(cfg)
    split_info = read_json(proc / "split_info.json")
    manifest: dict = {}

    for model in build_models(cfg):
        with mlflow.start_run(run_name=model.name) as run:
            mlflow.set_tags({"stage": "training", "model": model.name, "config_hash": config_hash(cfg),
                             "feature_version": data.store.meta["feature_version"]})
            mlflow.log_params({
                "model": model.name, **model.get_params(),
                "split_strategy": split_info["strategy"], "test_fraction": split_info["test_fraction"],
                "n_train_rows": split_info["n_train"], "n_train_users": data.R.shape[0], "n_items": data.R.shape[1],
                "positive_threshold": cfg["features"]["positive_threshold"], "seed": cfg["project"]["seed"],
            })
            start = time.time()
            model.fit(data)
            fit_seconds = time.time() - start
            metrics = {"fit_seconds": round(fit_seconds, 4)}
            if hasattr(model, "explained_variance"):
                metrics["explained_variance_ratio"] = model.explained_variance
                metrics["n_factors_effective"] = model.k_effective
            mlflow.log_metrics(metrics)

            path = Path(models_dir) / f"{model.name}.joblib"
            joblib.dump(model, path)
            mlflow.log_artifact(str(path), artifact_path="model")
            mlflow.log_artifact(cfg["_config_path"], artifact_path="config")
            catalog = rep / "feature_catalog.csv"
            if catalog.exists():
                mlflow.log_artifact(str(catalog), artifact_path="features")
            manifest[model.name] = {"run_id": run.info.run_id, "path": str(path.relative_to(cfg["_root"])),
                                    "params": model.get_params(), "fit_seconds": round(fit_seconds, 3)}
            log.info("Trained %-10s in %.2fs (MLflow run %s)", model.name, fit_seconds, run.info.run_id[:8])

    write_json(models_dir / "training_manifest.json", {"tracking_uri": tracking_uri, "models": manifest})
    return {"stage": "training", "finished_at": utc_now_iso(), "models": list(manifest),
            "tracking_uri": tracking_uri, "seconds": round(time.time() - t0, 2)}
