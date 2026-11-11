"""Stage 4 - Data transformation.

Turns the clean tables into model-ready derived datasets.  EVERYTHING here is computed
from the TRAINING split only, so no information from the test period leaks in.

Outputs (data/processed):
    train_transformed.parquet     ratings + centred / min-max normalised rating + positive label
    user_activity_summary.parquet per-user activity statistics
    item_popularity.parquet       per-item popularity statistics + rank
    interaction_matrix.npz        sparse user x item rating matrix  (+ matrix_index.npz id maps)
    item_genre_matrix.parquet     multi-hot encoding of item genres (+ genre_vocab.json)
    users_encoded.parquet         age scaling / bucket + gender code
reports/interaction_matrix_preview.csv  small dense excerpt for documentation
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import data_dir
from src.common.io import utc_now_iso, write_json
from src.common.log import get_logger

log = get_logger("transformation")


def build_genre_matrix(items: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    vocab = sorted({g for s in items["genres"] for g in s.split("|")})
    index = {g: j for j, g in enumerate(vocab)}
    mat = np.zeros((len(items), len(vocab)), dtype=np.int8)
    for r, s in enumerate(items["genres"]):
        for g in s.split("|"):
            mat[r, index[g]] = 1
    df = pd.DataFrame(mat, columns=[f"genre_{g}" for g in vocab])
    df.insert(0, "item_id", items["item_id"].to_numpy())
    return df, vocab


def build_interaction_matrix(train: pd.DataFrame, item_ids: np.ndarray):
    user_ids = np.sort(train["user_id"].unique())
    u_idx = np.searchsorted(user_ids, train["user_id"].to_numpy())
    i_idx = np.searchsorted(item_ids, train["item_id"].to_numpy())
    R = sp.csr_matrix((train["rating"].to_numpy(dtype="float32"), (u_idx, i_idx)),
                      shape=(len(user_ids), len(item_ids)))
    return R, user_ids


def run(cfg: dict) -> dict:
    t0 = time.time()
    proc, rep = data_dir(cfg, "processed"), data_dir(cfg, "reports")
    thr = cfg["features"]["positive_threshold"]
    rmin, rmax = cfg["validation"]["rating_min"], cfg["validation"]["rating_max"]

    train = pd.read_parquet(proc / "train.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet")
    users = pd.read_parquet(proc / "users_clean.parquet")
    as_of = int(train["timestamp"].max())

    # ---- 1. normalised ratings -------------------------------------------------------------------
    user_mean = train.groupby("user_id")["rating"].transform("mean")
    tt = train[["user_id", "item_id", "rating", "timestamp"]].copy()
    tt["rating_centered"] = tt["rating"] - user_mean
    tt["rating_minmax"] = (tt["rating"] - rmin) / (rmax - rmin)
    tt["is_positive"] = (tt["rating"] >= thr).astype("int8")
    tt.to_parquet(proc / "train_transformed.parquet", index=False)

    # ---- 2. user activity summary -----------------------------------------------------------------
    g = train.assign(day=train["timestamp"] // 86400).groupby("user_id")
    activity = pd.DataFrame({
        "n_interactions": g.size(), "avg_rating": g["rating"].mean(), "rating_std": g["rating"].std().fillna(0.0),
        "n_positive": g["rating"].apply(lambda s: int((s >= thr).sum())),
        "first_timestamp": g["timestamp"].min(), "last_timestamp": g["timestamp"].max(),
        "n_active_days": g["day"].nunique(),
    }).reset_index()
    activity["days_since_last_interaction"] = (as_of - activity["last_timestamp"]) / 86400.0
    activity.to_parquet(proc / "user_activity_summary.parquet", index=False)

    # ---- 3. item popularity ------------------------------------------------------------------------
    gi = train.groupby("item_id")
    pop = pd.DataFrame({"n_interactions": gi.size(), "avg_rating": gi["rating"].mean(),
                        "n_positive": gi["rating"].apply(lambda s: int((s >= thr).sum()))}).reset_index()
    pop["popularity_rank"] = pop["n_interactions"].rank(method="min", ascending=False).astype(int)
    pop["share_of_interactions"] = pop["n_interactions"] / pop["n_interactions"].sum()
    pop = pop.sort_values("popularity_rank").reset_index(drop=True)
    pop.to_parquet(proc / "item_popularity.parquet", index=False)

    # ---- 4. sparse user-item interaction matrix (catalog = ALL clean items) ------------------------------
    item_ids = np.sort(items["item_id"].to_numpy())
    R, user_ids = build_interaction_matrix(train, item_ids)
    sp.save_npz(proc / "interaction_matrix.npz", R)
    np.savez(proc / "matrix_index.npz", user_ids=user_ids, item_ids=item_ids)
    sparsity = 1 - R.nnz / (R.shape[0] * R.shape[1])
    preview = pd.DataFrame(R[:10, :10].toarray().astype(int), index=user_ids[:10], columns=item_ids[:10])
    preview.index.name = "user_id \\ item_id"
    preview.to_csv(rep / "interaction_matrix_preview.csv")

    # ---- 5. categorical encodings --------------------------------------------------------------------------
    genre_df, vocab = build_genre_matrix(items.set_index("item_id").loc[item_ids].reset_index())
    genre_df.to_parquet(proc / "item_genre_matrix.parquet", index=False)
    write_json(proc / "genre_vocab.json", vocab)

    fcfg = cfg["features"]
    enc = users[["user_id"]].copy()
    enc["age_scaled"] = (users["age"] - users["age"].min()) / max(users["age"].max() - users["age"].min(), 1)
    enc["age_group"] = pd.cut(users["age"], bins=fcfg["age_bins"], labels=fcfg["age_labels"], right=False).astype(str)
    enc["gender_code"] = users["gender"].map({"M": 0, "F": 1}).fillna(-1).astype("int8")
    enc.to_parquet(proc / "users_encoded.parquet", index=False)

    summary = {"stage": "transformation", "finished_at": utc_now_iso(),
               "matrix_shape": list(R.shape), "matrix_nnz": int(R.nnz), "matrix_sparsity_pct": round(100 * sparsity, 2),
               "n_genres": len(vocab), "n_users_train": int(len(user_ids)), "as_of_timestamp": as_of,
               "seconds": round(time.time() - t0, 2)}
    log.info("Matrix %s, sparsity %.2f%%, %d genres", R.shape, 100 * sparsity, len(vocab))
    return summary
