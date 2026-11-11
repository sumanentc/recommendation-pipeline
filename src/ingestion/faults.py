"""Reproducible fault injection - used ONLY to demonstrate the validation stage.

Real datasets such as MovieLens are almost perfectly clean, which would make the
validation report empty and boring.  We therefore corrupt a small, seeded fraction of
the raw data (clearly logged in ``data/raw/ingestion_manifest.json``) so that every
validation rule has something to catch.  Switch it off with ``ingestion.inject_faults: false``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inject_faults(users: pd.DataFrame, items: pd.DataFrame, inter: pd.DataFrame,
                  seed: int = 42, rate: float = 0.002):
    rng = np.random.default_rng(seed + 1)
    users = users.copy().astype(object)
    items = items.copy().astype(object)
    inter = inter.copy().astype(object)
    n = len(inter)
    k = max(5, int(n * rate))
    injected: dict[str, int] = {}

    # ---------------- interactions: disjoint row segments -------------------------------
    perm = rng.permutation(n)
    cursor = 0

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        seg = perm[cursor: cursor + count]
        cursor += count
        return seg

    seg = take(k); inter.iloc[seg, inter.columns.get_loc("rating")] = rng.choice([0, 6, 7, -1, 10], k)
    injected["bad_rating"] = k
    seg = take(k); inter.iloc[seg, inter.columns.get_loc("rating")] = np.nan
    injected["missing_rating"] = k
    seg = take(k); inter.iloc[seg, inter.columns.get_loc("rating")] = rng.choice(["five", "excellent", "good"], k)
    injected["str_rating"] = k
    seg = take(k); inter.iloc[seg, inter.columns.get_loc("timestamp")] = rng.choice([-1, 0], k)
    injected["bad_ts_neg"] = k
    seg = take(k); inter.iloc[seg, inter.columns.get_loc("timestamp")] = 4102444800   # year 2100
    injected["bad_ts_future"] = k
    seg = take(k); inter.iloc[seg, inter.columns.get_loc("timestamp")] = rng.choice(["not_a_timestamp", "yesterday"], k)
    injected["bad_ts_str"] = k
    seg = take(k); inter.iloc[seg, inter.columns.get_loc("user_id")] = int(users["user_id"].max()) + 1000 + np.arange(k)
    injected["orphan_user"] = k
    seg = take(k); inter.iloc[seg, inter.columns.get_loc("item_id")] = int(items["item_id"].max()) + 1000 + np.arange(k)
    injected["orphan_item"] = k
    seg = take(k); inter.iloc[seg, inter.columns.get_loc("user_id")] = np.nan
    injected["missing_user_id"] = k

    clean_positions = perm[cursor:]                      # rows we did not touch
    dup_src = rng.choice(clean_positions, k, replace=False)
    dups = inter.iloc[dup_src].copy()
    # valid re-ratings: same user+item, LATER timestamp, different rating (handled in preparation)
    rerate_src = rng.choice(np.setdiff1d(clean_positions, dup_src), k, replace=False)
    rerate = inter.iloc[rerate_src].copy()
    rerate["timestamp"] = rerate["timestamp"].astype(int) + rng.integers(86400, 86400 * 60, k)
    rerate["rating"] = ((rerate["rating"].astype(int) + rng.integers(1, 4, k) - 1) % 5) + 1
    inter = pd.concat([inter, dups, rerate], ignore_index=True)
    injected["duplicate_interaction"] = k
    injected["valid_re_ratings"] = k

    # ---------------- users ----------------------------------------------------------------
    u_idx = rng.permutation(len(users))
    uc = users.columns.get_loc
    users.iloc[u_idx[0:5], uc("age")] = np.nan
    users.iloc[u_idx[5:7], uc("age")] = 0
    users.iloc[u_idx[7:9], uc("age")] = 250
    users.iloc[u_idx[9:11], uc("age")] = "abc"
    users.iloc[u_idx[11:16], uc("gender")] = np.nan
    users.iloc[u_idx[16:18], uc("gender")] = "X"
    users.iloc[u_idx[18:21], uc("occupation")] = np.nan
    users = pd.concat([users, users.iloc[u_idx[21:24]]], ignore_index=True)
    injected.update(user_missing_age=5, user_age_out_of_range=4, user_age_not_numeric=2,
                    user_missing_gender=5, user_invalid_gender=2, user_missing_occupation=3,
                    user_duplicate=3)

    # ---------------- items ----------------------------------------------------------------
    i_idx = rng.permutation(len(items))
    ic = items.columns.get_loc
    items.iloc[i_idx[0:4], ic("title")] = np.nan
    items.iloc[i_idx[4:7], ic("release_date")] = "31-Foo-1990"
    items.iloc[i_idx[7:9], ic("release_date")] = "not a date"
    items = pd.concat([items, items.iloc[i_idx[9:12]]], ignore_index=True)
    injected.update(item_missing_title=4, item_invalid_release_date=5, item_duplicate=3)

    return users, items, inter, injected
