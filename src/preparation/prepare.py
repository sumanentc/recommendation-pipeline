"""Stage 3 - Data preparation.

* clean + standardise users / items / interactions
* handle missing values and duplicates
* TIME-BASED train/test split (no future information in the training set)

Outputs (data/processed): users_clean, items_clean, interactions_clean, train, test (parquet)
and split_info.json.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.common.config import data_dir
from src.common.io import utc_now_iso, write_json
from src.common.log import get_logger

log = get_logger("preparation")


def clean_users(users: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    u = users.copy()
    stats = {"age_imputed": int(u["age"].isna().sum())}
    u["age"] = u["age"].fillna(u["age"].median()).round().astype("int64")
    g = u["gender"].fillna("U").astype(str).str.strip().str.upper()
    stats["gender_set_unknown"] = int((~g.isin(["M", "F"])).sum())
    u["gender"] = g.where(g.isin(["M", "F"]), "U")
    occ = u["occupation"].fillna("unknown").astype(str).str.strip().str.lower()
    u["occupation"] = occ.replace("", "unknown")
    u["zip_code"] = u["zip_code"].fillna("unknown").astype(str).str.strip()
    u = u.drop_duplicates("user_id").sort_values("user_id").reset_index(drop=True)
    return u, stats


def clean_items(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    it = items.copy()
    stats = {"title_filled": int(it["title"].isna().sum()), "release_date_imputed": int(it["release_date"].isna().sum())}
    it["title"] = it["title"].fillna("Unknown title").astype(str).str.strip()
    year = it["release_date"].dt.year
    median_year = int(year.median()) if year.notna().any() else 1995
    it["release_year"] = year.fillna(median_year).astype("int64")

    def norm_genres(s) -> str:
        if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
            return "unknown"
        parts = sorted({p.strip().lower() for p in str(s).split("|") if p.strip()})
        return "|".join(parts) if parts else "unknown"

    it["genres"] = it["genres"].map(norm_genres)
    it = it.drop(columns=["release_date"]).drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)
    return it, stats


def clean_interactions(inter: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = inter.sort_values(["timestamp", "user_id", "item_id"], kind="stable")
    before = len(x)
    # A user may rate the same item more than once -> keep the LATEST rating (most up-to-date taste)
    x = x.drop_duplicates(["user_id", "item_id"], keep="last")
    stats = {"repeated_user_item_pairs_collapsed": before - len(x)}
    x = x.assign(rating=x["rating"].astype("float64"),
                 datetime=pd.to_datetime(x["timestamp"], unit="s", utc=True))
    return x.sort_values(["timestamp", "user_id", "item_id"], kind="stable").reset_index(drop=True), stats


def time_based_split(inter: pd.DataFrame, strategy: str, test_fraction: float, min_n: int):
    """Return boolean mask ``is_test``. Test rows are always LATER than train rows."""
    inter = inter.sort_values(["user_id", "timestamp", "item_id"], kind="stable")
    if strategy == "global":
        cutoff = inter["timestamp"].quantile(1 - test_fraction)
        is_test = inter["timestamp"] > cutoff
    elif strategy == "per_user":
        pos = inter.groupby("user_id").cumcount().to_numpy()
        n = inter.groupby("user_id")["item_id"].transform("size").to_numpy()
        n_test = np.where(n >= min_n, np.maximum(1, np.floor(n * test_fraction).astype(int)), 0)
        is_test = pd.Series(pos >= (n - n_test), index=inter.index)
    else:
        raise ValueError(f"unknown split_strategy: {strategy}")
    return inter, is_test


def run(cfg: dict) -> dict:
    t0 = time.time()
    proc = data_dir(cfg, "processed")
    pcfg = cfg["preparation"]

    users, u_stats = clean_users(pd.read_parquet(proc / "valid_users.parquet"))
    items, i_stats = clean_items(pd.read_parquet(proc / "valid_items.parquet"))
    inter, x_stats = clean_interactions(pd.read_parquet(proc / "valid_interactions.parquet"))

    users.to_parquet(proc / "users_clean.parquet", index=False)
    items.to_parquet(proc / "items_clean.parquet", index=False)
    inter.to_parquet(proc / "interactions_clean.parquet", index=False)

    ordered, is_test = time_based_split(inter, pcfg["split_strategy"], pcfg["test_fraction"],
                                        pcfg["min_user_interactions_for_test"])
    train = ordered[~is_test].sort_values(["timestamp", "user_id", "item_id"], kind="stable").reset_index(drop=True)
    test_all = ordered[is_test].sort_values(["timestamp", "user_id", "item_id"], kind="stable").reset_index(drop=True)

    # keep only test rows whose user AND item were seen in training ("warm" evaluation)
    known_u, known_i = set(train["user_id"]), set(train["item_id"])
    keep = test_all["user_id"].isin(known_u) & test_all["item_id"].isin(known_i)
    test = test_all[keep].reset_index(drop=True)

    # ---- leakage check: for every user, all train events are not later than the user's test events
    last_train = train.groupby("user_id")["timestamp"].max()
    first_test = test.groupby("user_id")["timestamp"].min()
    joined = pd.concat([last_train, first_test], axis=1, keys=["lt", "ft"]).dropna()
    if pcfg["split_strategy"] == "global":
        leak_ok = bool(train["timestamp"].max() <= test["timestamp"].min())
    else:
        leak_ok = bool((joined["lt"] <= joined["ft"]).all())
    if not leak_ok:
        raise AssertionError("Temporal leakage detected in train/test split")

    train.to_parquet(proc / "train.parquet", index=False)
    test.to_parquet(proc / "test.parquet", index=False)

    def span(df):
        return [str(df["datetime"].min()), str(df["datetime"].max())] if len(df) else None

    info = {
        "strategy": pcfg["split_strategy"], "test_fraction": pcfg["test_fraction"],
        "n_clean_interactions": len(inter), "n_train": len(train), "n_test": len(test),
        "n_test_dropped_cold_start": int(len(test_all) - len(test)),
        "n_train_users": int(train["user_id"].nunique()), "n_test_users": int(test["user_id"].nunique()),
        "n_items_catalog": len(items), "train_time_span": span(train), "test_time_span": span(test),
        "temporal_leakage_check": "passed",
    }
    write_json(proc / "split_info.json", info)
    log.info("Split (%s): train=%d test=%d (dropped %d cold-start test rows)",
             pcfg["split_strategy"], len(train), len(test), info["n_test_dropped_cold_start"])
    return {"stage": "preparation", "finished_at": utc_now_iso(), "users": u_stats, "items": i_stats,
            "interactions": x_stats, "split": info, "seconds": round(time.time() - t0, 2)}
