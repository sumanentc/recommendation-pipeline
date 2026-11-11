"""Evaluation metrics for top-K recommendation (binary relevance) and rating prediction."""
from __future__ import annotations

import numpy as np


def precision_recall_ndcg_at_k(recs: np.ndarray, relevant: list[set], k: int):
    """recs: (n_users, >=k) ranked item ids; relevant: list of sets of relevant item ids (one per user).

    Precision@K = hits / K
    Recall@K    = hits / |relevant|
    NDCG@K      = DCG@K / IDCG@K,  DCG = sum_i rel_i / log2(i + 1)   (i = 1-based rank)
    Returns per-user arrays (precision, recall, ndcg).
    """
    n = len(recs)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))             # ranks 1..k
    prec, rec, ndcg = np.zeros(n), np.zeros(n), np.zeros(n)
    for u in range(n):
        rel = relevant[u]
        hits = np.fromiter((item in rel for item in recs[u, :k]), dtype=float, count=k)
        n_hit = hits.sum()
        prec[u] = n_hit / k
        rec[u] = n_hit / len(rel) if rel else 0.0
        dcg = float((hits * discounts).sum())
        idcg = float(discounts[: min(len(rel), k)].sum())
        ndcg[u] = dcg / idcg if idcg > 0 else 0.0
    return prec, rec, ndcg


def catalog_coverage(recs: np.ndarray, n_catalog: int, k: int) -> float:
    """Share of the whole catalog that appears in at least one user's top-K list."""
    return float(len(np.unique(recs[:, :k])) / n_catalog)


def intra_list_diversity(recs: np.ndarray, item_ids: np.ndarray, item_genre: np.ndarray, k: int,
                         batch: int = 1000) -> float:
    """Average pairwise cosine DISTANCE (1 - cosine similarity) between genre vectors of recommended items.

    0 = all recommended items have identical genres, 1 = completely different genres.
    """
    norm = np.linalg.norm(item_genre, axis=1, keepdims=True)
    unit = item_genre / np.maximum(norm, 1e-12)
    pos = np.searchsorted(item_ids, recs[:, :k])
    vals = []
    for s in range(0, len(pos), batch):
        X = unit[pos[s:s + batch]]                                  # (b, k, g)
        S = np.einsum("bkg,blg->bkl", X, X)
        off_diag = (S.sum(axis=(1, 2)) - np.trace(S, axis1=1, axis2=2)) / (k * (k - 1))
        vals.append(1.0 - off_diag)
    return float(np.concatenate(vals).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
