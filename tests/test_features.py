import numpy as np
import pandas as pd

from src.common.config import data_dir
from src.features.store import FeatureStore
from src.pipeline import run_all


def test_feature_store_consistency_and_point_in_time(cfg):
    run_all(cfg, ["ingest", "validate", "prepare", "transform", "features"])
    proc, feat = data_dir(cfg, "processed"), data_dir(cfg, "features")
    train, test = pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "test.parquet")
    store = FeatureStore.load(feat, proc)

    # point in time: features are as of the last TRAIN timestamp; no feature counts test interactions
    assert store.meta["as_of_timestamp"] == int(train["timestamp"].max())
    uf = store.user_features
    assert (uf["user_activity_count"].loc[train["user_id"].unique()].to_numpy()
            == train.groupby("user_id").size().loc[train["user_id"].unique()].to_numpy()).all()
    assert store.item_features["item_popularity_count"].sum() == len(train)

    # training/prediction consistency: pair feature == entry of the vectorised affinity matrix
    u, i = test["user_id"].iloc[:50].to_numpy(), test["item_id"].iloc[:50].to_numpy()
    pf = store.pair_features(u, i)
    full = store.genre_affinity_matrix(u, i)                        # (users x items) -> take the diagonal
    assert np.allclose(pf["genre_affinity"].to_numpy(), np.diag(full), atol=1e-6)

    # cold-start user gets safe defaults instead of NaN / an exception
    cold = store.get_user_features([10_000_000])
    assert cold.loc[10_000_000, "user_activity_count"] == 0
    assert not cold[["user_avg_rating", "user_preferred_category"]].isna().any().any()
    # genre affinity rows are probability distributions
    assert np.allclose(store.affinity.sum(axis=1), 1.0, atol=1e-5)
