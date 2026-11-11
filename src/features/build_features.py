"""Stage 5 - Feature management: build + persist features and the feature catalog."""
from __future__ import annotations

import time

import pandas as pd

from src.common.config import data_dir
from src.common.io import read_json, utc_now_iso
from src.common.log import get_logger
from src.features.definitions import FEATURES, catalog_df
from src.features.store import FeatureStore

log = get_logger("features")


def write_catalog(rep_dir) -> pd.DataFrame:
    cat = catalog_df()
    cat.to_csv(rep_dir / "feature_catalog.csv", index=False)
    md = ["# Feature catalog", "",
          "All features are computed from the **training split only** (point-in-time correct) and are served by "
          "`src/features/store.py`, so training and prediction use identical code.", "",
          "| " + " | ".join(cat.columns) + " |", "|" + "|".join(["---"] * len(cat.columns)) + "|"]
    for _, r in cat.iterrows():
        md.append("| " + " | ".join(f"`{v}`" if c == "Feature" else str(v).replace("|", "\\|") for c, v in r.items()) + " |")
    (rep_dir / "feature_catalog.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return cat


def run(cfg: dict) -> dict:
    t0 = time.time()
    proc, feat, rep = data_dir(cfg, "processed"), data_dir(cfg, "features"), data_dir(cfg, "reports")

    train = pd.read_parquet(proc / "train.parquet")
    users = pd.read_parquet(proc / "users_clean.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet")
    genre_df = pd.read_parquet(proc / "item_genre_matrix.parquet")
    vocab = read_json(proc / "genre_vocab.json")

    store = FeatureStore.build(train, users, items, genre_df, vocab, cfg["features"])
    store.save(feat)

    # Materialise interaction (pair) features for the training rows - the SAME method is used at prediction time.
    pairs = store.pair_features(train["user_id"].to_numpy(), train["item_id"].to_numpy())
    pairs["rating"] = train["rating"].to_numpy()
    pairs["interaction_age_days"] = (store.meta["as_of_timestamp"] - train["timestamp"].to_numpy()) / 86400.0
    pairs.to_parquet(feat / "interaction_features.parquet", index=False)

    write_catalog(rep)
    log.info("Feature store v%s: %d user rows, %d item rows, %d pair rows; catalog has %d features",
             store.meta["feature_version"], len(store.user_features), len(store.item_features), len(pairs), len(FEATURES))
    return {"stage": "features", "finished_at": utc_now_iso(), "feature_version": store.meta["feature_version"],
            "n_user_rows": len(store.user_features), "n_item_rows": len(store.item_features),
            "n_pair_rows": len(pairs), "n_catalog_features": len(FEATURES), "seconds": round(time.time() - t0, 2)}
