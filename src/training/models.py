"""Recommender models.

PopularityRecommender  - baseline: recommend what is globally most popular (same list for everyone)
SVDRecommender         - personalised collaborative filtering via matrix factorisation (TruncatedSVD)
HybridRecommender      - SVD scores blended with a content score (genre affinity from the feature store)

All models share the same interface:
    fit(data)                          learn from a TrainingData object
    score(user_ids)  -> (n_users, n_items) matrix, higher = better   (unknown users -> popularity fallback)
    recommend(user_ids, k)             top-k item ids, already-seen items removed
    predict_rating(user_ids, item_ids) explicit rating prediction (for RMSE / MAE)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from src.features.store import FeatureStore


@dataclass
class TrainingData:
    R: sp.csr_matrix              # user x item ratings (train split)
    user_ids: np.ndarray          # row ids of R   (sorted)
    item_ids: np.ndarray          # column ids of R (full catalog, sorted)
    store: FeatureStore
    rating_min: float = 1.0
    rating_max: float = 5.0


class BaseRecommender:
    name = "base"

    def get_params(self) -> dict:
        return {}

    # ------------------------------------------------------------------------------------ fitting
    def fit(self, data: TrainingData) -> "BaseRecommender":
        self.R = data.R.tocsr()
        self.user_ids, self.item_ids = data.user_ids, data.item_ids
        self.store = data.store
        self.rmin, self.rmax = data.rating_min, data.rating_max
        self._u_index = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids)
        self._i_index = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)
        itf = self.store.get_item_features(self.item_ids)
        # popularity from the FEATURE STORE; tiny rating-based tie-breaker
        self.pop_vector = (itf["item_popularity_count"].to_numpy(float)
                           + 1e-3 * itf["item_bayes_avg_rating"].to_numpy(float)).astype("float32")
        self.item_mean = itf["item_bayes_avg_rating"].to_numpy(float)
        self.global_mean = float(self.store.meta["global_mean"])
        counts = np.diff(self.R.indptr)
        sums = np.asarray(self.R.sum(axis=1)).ravel()
        self.user_mean = np.where(counts > 0, sums / np.maximum(counts, 1), self.global_mean)
        self._fit(data)
        return self

    def _fit(self, data: TrainingData) -> None:  # pragma: no cover - overridden
        pass

    def _score_known(self, rows: np.ndarray) -> np.ndarray:
        return np.tile(self.pop_vector, (len(rows), 1))

    # ------------------------------------------------------------------------------------ prediction
    def _rows(self, user_ids) -> np.ndarray:
        return self._u_index.reindex(np.asarray(user_ids)).fillna(-1).astype(int).to_numpy()

    def score(self, user_ids) -> np.ndarray:
        rows = self._rows(user_ids)
        out = np.tile(self.pop_vector, (len(rows), 1)).astype("float32")      # cold-start fallback = popularity
        known = rows >= 0
        if known.any():
            out[known] = self._score_known(rows[known])
        return out

    def recommend(self, user_ids, k: int = 10, exclude_seen: bool = True, batch_size: int = 512) -> np.ndarray:
        user_ids = np.asarray(user_ids)
        result = np.empty((len(user_ids), k), dtype=self.item_ids.dtype)
        for start in range(0, len(user_ids), batch_size):
            batch = user_ids[start:start + batch_size]
            scores = self.score(batch)
            if exclude_seen:
                rows = self._rows(batch)
                for b, r in enumerate(rows):
                    if r >= 0:
                        seen = self.R.indices[self.R.indptr[r]:self.R.indptr[r + 1]]
                        scores[b, seen] = -np.inf
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            result[start:start + batch_size] = self.item_ids[np.take_along_axis(top, order, axis=1)]
        return result

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        """Default rating head: Bayesian-damped item mean (fallback: global mean)."""
        cols = self._i_index.reindex(np.asarray(item_ids)).fillna(-1).astype(int).to_numpy()
        pred = np.where(cols >= 0, self.item_mean[np.maximum(cols, 0)], self.global_mean)
        return np.clip(pred, self.rmin, self.rmax)


class PopularityRecommender(BaseRecommender):
    """Baseline. Everybody gets the globally most-interacted-with items they have not seen yet."""
    name = "popularity"


def _als_biased_mf(R: sp.csr_matrix, k: int, reg: float, iters: int, seed: int):
    """Biased matrix factorisation for explicit ratings, fitted by alternating least squares (ALS).

        r_hat(u, i) = mu + b_u + b_i + p_u . q_i

    Each half-step solves one small ridge regression per user (or per item) in closed form.
    Only OBSERVED ratings are used (missing ratings are not treated as zeros).
    """
    rng = np.random.default_rng(seed)
    n_u, n_i = R.shape
    Rc = R.tocsc()
    mu = float(R.data.mean())
    P, Q = rng.normal(0, 0.1, (n_u, k)), rng.normal(0, 0.1, (n_i, k))
    bu, bi = np.zeros(n_u), np.zeros(n_i)
    eye = reg * np.eye(k + 1)

    def solve(indptr, indices, data, n, other, other_bias, own_bias, own_vec):
        for r in range(n):
            s, e = indptr[r], indptr[r + 1]
            if s == e:
                continue
            idx = indices[s:e]
            X = np.hstack([np.ones((e - s, 1)), other[idx]])
            y = data[s:e] - mu - other_bias[idx]
            w = np.linalg.solve(X.T @ X + eye, X.T @ y)
            own_bias[r], own_vec[r] = w[0], w[1:]

    for _ in range(iters):
        solve(R.indptr, R.indices, R.data, n_u, Q, bi, bu, P)       # update users (items fixed)
        solve(Rc.indptr, Rc.indices, Rc.data, n_i, P, bu, bi, Q)    # update items (users fixed)
    return mu, bu, bi, P, Q


class SVDRecommender(BaseRecommender):
    """Personalised matrix factorisation with two heads.

    * ranking head : TruncatedSVD of the raw rating matrix (a.k.a. PureSVD) -> scores = U_k S_k V_k^T.
                     Used for top-N lists (it also learns *which* items users tend to interact with).
    * rating head  : biased matrix factorisation fitted by ALS -> r_hat = mu + b_u + b_i + p_u . q_i.
                     Used for RMSE / MAE (rating prediction is a different task than ranking).
    """
    name = "svd"

    def __init__(self, n_factors: int = 32, rating_factors: int = 16, als_reg: float = 10.0,
                 als_iters: int = 12, seed: int = 42):
        self.n_factors, self.rating_factors = n_factors, rating_factors
        self.als_reg, self.als_iters, self.seed = als_reg, als_iters, seed

    def get_params(self) -> dict:
        return {"n_factors": self.n_factors, "rating_factors": self.rating_factors,
                "als_reg": self.als_reg, "als_iters": self.als_iters, "seed": self.seed}

    def _fit(self, data: TrainingData) -> None:
        k = int(max(2, min(self.n_factors, min(self.R.shape) - 1)))
        self.k_effective = k
        svd = TruncatedSVD(n_components=k, random_state=self.seed, n_iter=7)
        self.U = svd.fit_transform(self.R).astype("float32")          # users x k  (already scaled by S)
        self.V = svd.components_.astype("float32")                    # k x items
        self.explained_variance = float(svd.explained_variance_ratio_.sum())
        self.mu, self.bu, self.bi, self.P, self.Q = _als_biased_mf(
            self.R, self.rating_factors, self.als_reg, self.als_iters, self.seed)

    def _score_known(self, rows: np.ndarray) -> np.ndarray:
        return self.U[rows] @ self.V

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        rows = self._rows(user_ids)
        cols = self._i_index.reindex(np.asarray(item_ids)).fillna(-1).astype(int).to_numpy()
        u_ok, i_ok = rows >= 0, cols >= 0
        pred = np.full(len(rows), self.mu)
        pred[u_ok] += self.bu[rows[u_ok]]                              # unknown user -> no user bias
        pred[i_ok] += self.bi[cols[i_ok]]                              # unknown item -> no item bias
        both = u_ok & i_ok
        pred[both] += np.einsum("ij,ij->i", self.P[rows[both]], self.Q[cols[both]])
        return np.clip(pred, self.rmin, self.rmax)


class HybridRecommender(SVDRecommender):
    """SVD collaborative score + genre-affinity content score (taken from the feature store).

    score = (1 - w) * z(cf_score) + w * z(content_score)      z = per-user standardisation
    """
    name = "hybrid"

    def __init__(self, n_factors: int = 32, content_weight: float = 0.2, rating_factors: int = 16,
                 als_reg: float = 10.0, als_iters: int = 12, seed: int = 42):
        super().__init__(n_factors, rating_factors, als_reg, als_iters, seed)
        self.content_weight = content_weight

    def get_params(self) -> dict:
        return {**super().get_params(), "content_weight": self.content_weight}

    @staticmethod
    def _z(m: np.ndarray) -> np.ndarray:
        return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)

    def _score_known(self, rows: np.ndarray) -> np.ndarray:
        cf = super()._score_known(rows)
        content = self.store.genre_affinity_matrix(self.user_ids[rows], self.item_ids)
        w = self.content_weight
        return ((1 - w) * self._z(cf) + w * self._z(content)).astype("float32")
