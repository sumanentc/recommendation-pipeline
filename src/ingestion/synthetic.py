"""Synthetic MovieLens-like data generator (used when offline or when requested).

The generator is deliberately *structured*, not random noise: users have latent genre
tastes, items have popularity + quality, and ratings depend on taste, so a personalised
model really can beat a popularity baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = ["action", "adventure", "animation", "comedy", "crime", "documentary", "drama",
          "fantasy", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
OCCUPATIONS = ["student", "engineer", "educator", "artist", "administrator", "writer",
               "technician", "librarian", "programmer", "other", "retired", "healthcare"]


def generate_synthetic(n_users: int, n_items: int, n_interactions: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_g = len(GENRES)

    # ---- items ------------------------------------------------------------------
    item_ids = np.arange(1, n_items + 1)
    item_genre = np.zeros((n_items, n_g))
    for i in range(n_items):
        k = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        item_genre[i, rng.choice(n_g, size=k, replace=False)] = 1
    pop_rank = rng.permutation(n_items) + 1
    popularity = 1.0 / pop_rank ** 0.8                      # Zipf-like long tail
    quality = rng.normal(0, 0.5, n_items)
    years = rng.integers(1950, 2021, n_items)
    items = pd.DataFrame({
        "item_id": item_ids,
        "title": [f"Movie {i:04d} ({years[i - 1]})" for i in item_ids],
        "release_date": [f"01-Jan-{y}" for y in years],
        "genres": ["|".join(GENRES[g] for g in np.flatnonzero(row)) for row in item_genre],
    })

    # ---- users ------------------------------------------------------------------
    user_ids = np.arange(1, n_users + 1)
    taste = rng.dirichlet(np.full(n_g, 0.25), size=n_users)  # peaky genre preferences
    taste = taste / taste.max(axis=1, keepdims=True)
    users = pd.DataFrame({
        "user_id": user_ids,
        "age": np.clip(rng.normal(34, 12, n_users).round(), 13, 70).astype(int),
        "gender": rng.choice(["M", "F"], n_users, p=[0.6, 0.4]),
        "occupation": rng.choice(OCCUPATIONS, n_users),
        "zip_code": [f"{z:05d}" for z in rng.integers(1000, 99999, n_users)],
    })
    user_bias = rng.normal(0, 0.35, n_users)

    # ---- interactions -------------------------------------------------------------
    t0 = int(pd.Timestamp("2022-01-01").timestamp())
    t1 = int(pd.Timestamp("2024-01-01").timestamp())
    per_user = rng.lognormal(mean=0.0, sigma=0.8, size=n_users)
    per_user = np.maximum(20, np.round(per_user / per_user.sum() * n_interactions)).astype(int)
    per_user = np.minimum(per_user, n_items // 2)

    rows = []
    affinity_all = (taste @ item_genre.T) / item_genre.sum(axis=1)[None, :]   # users x items
    for u in range(n_users):
        w = popularity * (0.15 + affinity_all[u] / affinity_all[u].max()) ** 2
        w = w / w.sum()
        chosen = rng.choice(n_items, size=per_user[u], replace=False, p=w)
        join = t0 + rng.integers(0, (t1 - t0) // 2)
        ts = np.sort(join + rng.integers(0, t1 - join, per_user[u]))
        aff = affinity_all[u, chosen] / affinity_all[u].max()
        score = 3.0 + 1.6 * (aff - aff.mean()) * 2 + quality[chosen] + user_bias[u] + rng.normal(0, 0.6, per_user[u])
        rating = np.clip(np.round(score), 1, 5).astype(int)
        for i, r, t in zip(chosen, rating, ts):
            rows.append((user_ids[u], item_ids[i], int(r), int(t)))
    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    interactions = interactions.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return users, items, interactions
