"""Feature registry = the single source of truth for the FEATURE CATALOG.

Each feature is *declared* once here (name, entity, type, meaning, logic, source, consumers).
`catalog.py` renders this registry as reports/feature_catalog.{md,csv}, and the feature
store (store.py) implements exactly these features - so documentation and code cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass(frozen=True)
class FeatureDef:
    name: str
    entity: str          # user | item | interaction
    dtype: str
    description: str
    logic: str
    source: str
    used_by: str


_U, _I, _X = "user", "item", "interaction"

FEATURES: list[FeatureDef] = [
    # ------------------------------------------------------------------ user features
    FeatureDef("user_activity_count", _U, "int", "How many items the user interacted with (train period).",
               "count(interactions) per user", "train", "cold-start detection (== 0), analysis"),
    FeatureDef("user_avg_rating", _U, "float", "The user's average given rating (rating generosity).",
               "mean(rating) per user; global mean for cold-start users", "train", "pair feature avg_rating_gap, analysis"),
    FeatureDef("user_rating_std", _U, "float", "Spread of the user's ratings.",
               "std(rating) per user", "train", "analysis"),
    FeatureDef("user_positive_rate", _U, "float", "Share of the user's ratings that are 'liked' (>= threshold).",
               "mean(rating >= 4) per user", "train", "analysis"),
    FeatureDef("user_active_days", _U, "int", "Number of distinct calendar days with activity.",
               "nunique(floor(timestamp / 86400)) per user", "train", "analysis"),
    FeatureDef("days_since_last_interaction", _U, "float", "Recency of the user (days).",
               "(as_of - max(timestamp)) / 86400, as_of = last timestamp in train", "train", "analysis, cold-start rules"),
    FeatureDef("user_preferred_category", _U, "category", "The genre the user liked most.",
               "argmax of positive-interaction counts per genre (all interactions if no positives)", "train + items", "pair feature preferred_category_match, explanations"),
    FeatureDef("user_preferred_category_share", _U, "float", "How dominant the preferred genre is (0-1).",
               "max genre affinity", "train + items", "analysis"),
    FeatureDef("user_age", _U, "int", "Age of the user (imputed with median if missing).",
               "users_clean.age", "users", "analysis"),
    FeatureDef("user_age_group", _U, "category", "Age bucket.",
               "pd.cut(age, bins)", "users", "analysis"),
    FeatureDef("user_gender", _U, "category", "Gender (M / F / U = unknown).",
               "users_clean.gender", "users", "analysis"),
    FeatureDef("user_occupation", _U, "category", "Occupation (lower-cased, 'unknown' if missing).",
               "users_clean.occupation", "users", "analysis"),
    FeatureDef("user_genre_affinity[g]", _U, "float vector", "Distribution over genres of the user's liked items (sums to 1).",
               "L1-normalised (positive user x item matrix @ item x genre matrix)", "train + items", "Hybrid (content score)"),
    # ------------------------------------------------------------------ item features
    FeatureDef("item_popularity_count", _I, "int", "Number of interactions the item received.",
               "count(interactions) per item", "train", "Popularity model"),
    FeatureDef("item_popularity_rank", _I, "int", "Rank by popularity (1 = most popular).",
               "rank(item_popularity_count, descending, method=min)", "train", "analysis"),
    FeatureDef("item_popularity_score", _I, "float", "Popularity scaled to 0-1, damped by a log.",
               "log1p(count) / log1p(max count)", "train", "analysis"),
    FeatureDef("item_avg_rating", _I, "float", "Mean rating of the item (global mean if never rated).",
               "mean(rating) per item", "train", "pair feature avg_rating_gap"),
    FeatureDef("item_bayes_avg_rating", _I, "float", "Mean rating shrunk to the global mean for items with few ratings.",
               "(C * global_mean + sum(rating)) / (C + count), C = 10", "train", "Popularity model (rating prediction + tie-break)"),
    FeatureDef("item_rating_std", _I, "float", "Disagreement between users about the item.",
               "std(rating) per item", "train", "analysis"),
    FeatureDef("item_positive_rate", _I, "float", "Share of ratings that are 'liked'.",
               "mean(rating >= 4) per item", "train", "analysis"),
    FeatureDef("item_days_since_last_interaction", _I, "float", "How recently the item was interacted with.",
               "(as_of - max(timestamp)) / 86400", "train", "analysis"),
    FeatureDef("item_age_years", _I, "int", "Age of the item at as_of time.",
               "as_of year - release_year", "items", "analysis"),
    FeatureDef("item_n_genres", _I, "int", "Number of genres of the item.",
               "len(genres)", "items", "genre_affinity normalisation"),
    FeatureDef("item_primary_genre", _I, "category", "First genre (alphabetical) of the item.",
               "sorted(genres)[0]", "items", "analysis"),
    FeatureDef("item_genres", _I, "string", "All genres, '|' separated.",
               "items_clean.genres", "items", "genre matrix (Hybrid, Diversity metric), explanations"),
    # ------------------------------------------------------------------ interaction (user-item pair) features
    FeatureDef("genre_affinity", _X, "float", "How well the item's genres match the user's genre taste.",
               "mean over the item's genres of user_genre_affinity[g]", "user + item features", "Hybrid (computed at training AND prediction time)"),
    FeatureDef("preferred_category_match", _X, "int (0/1)", "1 if the item belongs to the user's preferred genre.",
               "user_preferred_category in item_genres", "user + item features", "analysis, explanations"),
    FeatureDef("avg_rating_gap", _X, "float", "Is the item better or worse rated than the user's usual rating?",
               "item_avg_rating - user_avg_rating", "user + item features", "analysis"),
    FeatureDef("interaction_age_days", _X, "float", "How old an observed interaction is (train rows only).",
               "(as_of - timestamp) / 86400", "train", "recency weighting (future work)"),
]


def catalog_df() -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in FEATURES]).rename(columns={
        "name": "Feature", "entity": "Entity", "dtype": "Type", "description": "Description",
        "logic": "Computation logic", "source": "Source", "used_by": "Used by"})
