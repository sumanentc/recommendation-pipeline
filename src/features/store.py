"""A small, custom FEATURE STORE (offline store = parquet files, online lookup = pandas).

Why custom instead of Feast?  The project is small and Feast adds heavy infrastructure; the
assignment explicitly allows "Feast or a custom feature layer".  The important MLOps
properties are kept:

* one implementation of every feature (training and prediction call the SAME methods)
* point-in-time correctness: features are computed from the TRAIN split only, with
  ``as_of`` = last training timestamp, so nothing from the test period leaks in
* versioned + documented (feature_metadata.json, feature catalog)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.io import read_json, utc_now_iso, write_json

COLD_USER_DEFAULTS = {
    "user_activity_count": 0, "user_rating_std": 0.0, "user_positive_rate": 0.0, "user_active_days": 0,
    "user_preferred_category": "none", "user_preferred_category_share": 0.0,
    "user_age_group": "unknown", "user_gender": "U", "user_occupation": "unknown",
}


class FeatureStore:
    def __init__(self, user_features: pd.DataFrame, item_features: pd.DataFrame, affinity: pd.DataFrame,
                 item_genre: pd.DataFrame, meta: dict):
        self.user_features = user_features        # index = user_id
        self.item_features = item_features        # index = item_id (full catalog)
        self.affinity = affinity                  # index = user_id, columns = genre vocab
        self.item_genre = item_genre              # index = item_id, columns = genre vocab (0/1)
        self.meta = meta
        self.vocab: list[str] = list(item_genre.columns)
        self._vocab_index = {g: j for j, g in enumerate(self.vocab)}
        self._global_mean = float(meta["global_mean"])

    # ------------------------------------------------------------------------------------------ build
    @classmethod
    def build(cls, train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame,
              genre_df: pd.DataFrame, vocab: list[str], fcfg: dict) -> "FeatureStore":
        thr, prior = fcfg["positive_threshold"], fcfg["bayes_prior"]
        as_of = int(train["timestamp"].max())
        as_of_year = int(pd.to_datetime(as_of, unit="s").year)
        gmean = float(train["rating"].mean())

        item_ids = genre_df["item_id"].to_numpy()
        G = genre_df.drop(columns="item_id").to_numpy(dtype="float32")             # items x genres
        item_genre = pd.DataFrame(G, index=item_ids, columns=vocab)

        # ---------------------------------------------------------------- user genre affinity
        user_ids_tr = np.sort(train["user_id"].unique())
        u_idx = np.searchsorted(user_ids_tr, train["user_id"].to_numpy())
        i_idx = np.searchsorted(item_ids, train["item_id"].to_numpy())
        pos = (train["rating"] >= thr).to_numpy()
        shape = (len(user_ids_tr), len(item_ids))
        P = sp.csr_matrix((np.ones(pos.sum(), dtype="float32"), (u_idx[pos], i_idx[pos])), shape=shape)
        A_all = sp.csr_matrix((np.ones(len(train), dtype="float32"), (u_idx, i_idx)), shape=shape)
        C_pos, C_all = np.asarray(P @ G), np.asarray(A_all @ G)
        C = np.where((C_pos.sum(1) > 0)[:, None], C_pos, C_all)
        aff = C / np.maximum(C.sum(1, keepdims=True), 1e-12)
        affinity = pd.DataFrame(aff, index=user_ids_tr, columns=vocab)
        pref = pd.Series(np.array(vocab, dtype=object)[C.argmax(1)], index=user_ids_tr)

        # ---------------------------------------------------------------- user features
        g = train.assign(day=train["timestamp"] // 86400, pos=(train["rating"] >= thr).astype(float)).groupby("user_id")
        uf = pd.DataFrame({
            "user_activity_count": g.size(), "user_avg_rating": g["rating"].mean(),
            "user_rating_std": g["rating"].std().fillna(0.0), "user_positive_rate": g["pos"].mean(),
            "user_active_days": g["day"].nunique(), "_last_ts": g["timestamp"].max(),
        })
        uf["days_since_last_interaction"] = (as_of - uf.pop("_last_ts")) / 86400.0
        uf["user_preferred_category"] = pref
        uf["user_preferred_category_share"] = aff.max(1) if len(aff) else []
        demo = users.set_index("user_id")
        uf = uf.reindex(demo.index.union(uf.index))
        uf["user_age"] = demo["age"]
        uf["user_age_group"] = pd.cut(demo["age"], bins=fcfg["age_bins"], labels=fcfg["age_labels"], right=False).astype(str)
        uf["user_gender"], uf["user_occupation"] = demo["gender"], demo["occupation"]
        uf["user_avg_rating"] = uf["user_avg_rating"].fillna(gmean)
        uf["user_age"] = uf["user_age"].fillna(users["age"].median())
        uf = uf.fillna(COLD_USER_DEFAULTS)
        uf.index.name = "user_id"

        # ---------------------------------------------------------------- item features
        gi = train.assign(pos=(train["rating"] >= thr).astype(float)).groupby("item_id")
        itf = pd.DataFrame({
            "item_popularity_count": gi.size(), "item_avg_rating": gi["rating"].mean(),
            "item_rating_std": gi["rating"].std(), "item_positive_rate": gi["pos"].mean(),
            "_last_ts": gi["timestamp"].max(),
        }).reindex(item_ids)
        cnt = itf["item_popularity_count"].fillna(0)
        itf["item_popularity_count"] = cnt.astype("int64")
        itf["item_avg_rating"] = itf["item_avg_rating"].fillna(gmean)
        itf["item_rating_std"] = itf["item_rating_std"].fillna(0.0)
        itf["item_positive_rate"] = itf["item_positive_rate"].fillna(0.0)
        itf["item_bayes_avg_rating"] = (prior * gmean + itf["item_avg_rating"] * cnt) / (prior + cnt)
        itf["item_popularity_rank"] = cnt.rank(method="min", ascending=False).astype("int64")
        itf["item_popularity_score"] = np.log1p(cnt) / np.log1p(max(cnt.max(), 1))
        itf["item_days_since_last_interaction"] = (as_of - itf.pop("_last_ts")) / 86400.0
        meta_items = items.set_index("item_id").loc[item_ids]
        itf["item_age_years"] = (as_of_year - meta_items["release_year"]).clip(lower=0).astype("int64")
        n_g = G.sum(1).astype(int)
        itf["item_n_genres"] = n_g
        itf["item_primary_genre"] = np.array(vocab, dtype=object)[G.argmax(1)]
        itf["item_genres"] = meta_items["genres"].to_numpy()
        itf.index.name = "item_id"

        version = hashlib.sha256(
            json.dumps([as_of, len(uf), len(itf), thr, prior, round(gmean, 6)]).encode()).hexdigest()[:10]
        meta = {"feature_version": version, "built_at": utc_now_iso(), "as_of_timestamp": as_of,
                "as_of_datetime": str(pd.to_datetime(as_of, unit="s", utc=True)), "positive_threshold": thr,
                "bayes_prior": prior, "global_mean": gmean, "n_users": int(len(uf)), "n_items": int(len(itf)),
                "n_users_with_history": int(len(user_ids_tr)), "genre_vocab": vocab}
        return cls(uf, itf, affinity, item_genre, meta)

    # ------------------------------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        self.user_features.reset_index().to_parquet(directory / "user_features.parquet", index=False)
        self.item_features.reset_index().to_parquet(directory / "item_features.parquet", index=False)
        self.affinity.reset_index(names="user_id").to_parquet(directory / "user_genre_affinity.parquet", index=False)
        write_json(directory / "feature_metadata.json", self.meta)

    @classmethod
    def load(cls, features_dir: Path, processed_dir: Path) -> "FeatureStore":
        features_dir, processed_dir = Path(features_dir), Path(processed_dir)
        meta = read_json(features_dir / "feature_metadata.json")
        uf = pd.read_parquet(features_dir / "user_features.parquet").set_index("user_id")
        itf = pd.read_parquet(features_dir / "item_features.parquet").set_index("item_id")
        aff = pd.read_parquet(features_dir / "user_genre_affinity.parquet").set_index("user_id")
        gdf = pd.read_parquet(processed_dir / "item_genre_matrix.parquet")
        vocab = read_json(processed_dir / "genre_vocab.json")
        item_genre = pd.DataFrame(gdf.drop(columns="item_id").to_numpy(dtype="float32"),
                                  index=gdf["item_id"].to_numpy(), columns=vocab)
        return cls(uf, itf, aff, item_genre, meta)

    # ------------------------------------------------------------------------------------------ online lookup
    def get_user_features(self, user_ids) -> pd.DataFrame:
        out = self.user_features.reindex(np.asarray(user_ids))
        out = out.fillna({**COLD_USER_DEFAULTS, "user_avg_rating": self._global_mean})
        return out

    def get_item_features(self, item_ids) -> pd.DataFrame:
        out = self.item_features.reindex(np.asarray(item_ids))
        return out.fillna({"item_popularity_count": 0, "item_avg_rating": self._global_mean,
                           "item_bayes_avg_rating": self._global_mean, "item_popularity_score": 0.0,
                           "item_rating_std": 0.0, "item_positive_rate": 0.0, "item_n_genres": 0})

    def genre_affinity_matrix(self, user_ids, item_ids=None) -> np.ndarray:
        """users x items matrix of 'genre_affinity' (the same quantity as the pair feature)."""
        A = self.affinity.reindex(np.asarray(user_ids)).fillna(0.0).to_numpy(dtype="float32")
        genre = self.item_genre if item_ids is None else self.item_genre.reindex(np.asarray(item_ids)).fillna(0.0)
        G = genre.to_numpy(dtype="float32")
        return (A @ G.T) / np.maximum(G.sum(1), 1.0)[None, :]

    def pair_features(self, user_ids, item_ids) -> pd.DataFrame:
        """Interaction features for aligned (user_id, item_id) pairs - works for train AND prediction."""
        user_ids, item_ids = np.asarray(user_ids), np.asarray(item_ids)
        uf, itf = self.get_user_features(user_ids), self.get_item_features(item_ids)
        A = self.affinity.reindex(user_ids).fillna(0.0).to_numpy(dtype="float32")
        G = self.item_genre.reindex(item_ids).fillna(0.0).to_numpy(dtype="float32")
        genre_aff = (A * G).sum(1) / np.maximum(G.sum(1), 1.0)
        pref_idx = uf["user_preferred_category"].map(self._vocab_index).fillna(-1).astype(int).to_numpy()
        match = np.where(pref_idx >= 0, G[np.arange(len(G)), np.maximum(pref_idx, 0)], 0.0)
        return pd.DataFrame({
            "user_id": user_ids, "item_id": item_ids, "genre_affinity": genre_aff,
            "preferred_category_match": match.astype(int),
            "avg_rating_gap": itf["item_avg_rating"].to_numpy() - uf["user_avg_rating"].to_numpy(),
        })
