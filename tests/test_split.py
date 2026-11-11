import pandas as pd
import pytest

from src.common.config import data_dir
from src.ingestion import ingest
from src.preparation import prepare
from src.validation import validate


def _prepare(cfg, strategy):
    cfg["preparation"]["split_strategy"] = strategy
    ingest.run(cfg)
    validate.run(cfg)
    prepare.run(cfg)
    proc = data_dir(cfg, "processed")
    return pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "test.parquet")


def test_per_user_split_is_chronological_and_disjoint(cfg):
    train, test = _prepare(cfg, "per_user")
    last_train = train.groupby("user_id")["timestamp"].max()
    first_test = test.groupby("user_id")["timestamp"].min()
    both = pd.concat([last_train, first_test], axis=1, keys=["lt", "ft"]).dropna()
    assert (both["lt"] <= both["ft"]).all()
    assert set(test["user_id"]) <= set(train["user_id"])
    assert set(test["item_id"]) <= set(train["item_id"])
    pairs_train = set(zip(train["user_id"], train["item_id"]))
    assert not any(p in pairs_train for p in zip(test["user_id"], test["item_id"]))
    assert 0.1 < len(test) / (len(train) + len(test)) < 0.3


def test_global_split_uses_one_cutoff(cfg):
    train, test = _prepare(cfg, "global")
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_repeated_ratings_keep_latest(cfg):
    ingest.run(cfg)
    validate.run(cfg)
    prepare.run(cfg)
    clean = pd.read_parquet(data_dir(cfg, "processed") / "interactions_clean.parquet")
    assert not clean.duplicated(["user_id", "item_id"]).any()


def test_unknown_strategy_is_rejected(cfg):
    with pytest.raises(ValueError):
        _prepare(cfg, "random")
