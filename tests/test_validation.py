import pandas as pd
import pytest

from src.common.config import data_dir
from src.common.io import read_json
from src.ingestion import ingest
from src.validation import validate
from src.validation.validate import DataQualityError

# rules whose detected count must be >= the number of deliberately injected faults
STRICT = ["missing_user_id", "missing_rating", "invalid_type_rating", "rating_out_of_range", "invalid_type_timestamp",
          "timestamp_out_of_range", "duplicate_interaction", "orphan_user_id", "orphan_item_id", "duplicate_user_id",
          "missing_age", "age_out_of_range", "invalid_type_age", "missing_gender", "invalid_gender",
          "missing_occupation", "duplicate_item_id", "missing_title", "invalid_release_date"]


def test_every_injected_fault_is_detected(cfg):
    ingest.run(cfg)
    validate.run(cfg)
    report = read_json(data_dir(cfg, "reports") / "validation_report.json")
    by_rule = {(r["table"], r["rule"]): r for r in report["rules"]}
    checked = 0
    for (table, rule), r in by_rule.items():
        if rule in STRICT and r["injected"] is not None:
            assert r["failed"] >= r["injected"], f"{table}.{rule}: detected {r['failed']} < injected {r['injected']}"
            checked += 1
    assert checked >= 15


def test_valid_plus_rejected_equals_input_and_valid_data_is_clean(cfg):
    ingest.run(cfg)
    validate.run(cfg)
    proc, err = data_dir(cfg, "processed"), data_dir(cfg, "errors")
    report = read_json(data_dir(cfg, "reports") / "validation_report.json")
    for table, stats in report["tables"].items():
        assert stats["rows_valid"] + stats["rows_rejected"] == stats["rows_in"]
        assert len(pd.read_csv(err / f"rejected_{table}.csv")) == stats["rows_rejected"]
    x = pd.read_parquet(proc / "valid_interactions.parquet")
    users = pd.read_parquet(proc / "valid_users.parquet")
    items = pd.read_parquet(proc / "valid_items.parquet")
    assert x["rating"].between(1, 5).all()
    assert x["user_id"].isin(users["user_id"]).all() and x["item_id"].isin(items["item_id"]).all()
    assert not x.duplicated(["user_id", "item_id", "timestamp"]).any()
    assert x.notna().all().all()


def test_quality_gate_stops_the_pipeline(cfg):
    cfg["validation"]["max_error_rate"] = 0.0005
    ingest.run(cfg)
    with pytest.raises(DataQualityError):
        validate.run(cfg)
    # the error dataset is still written so that the problem can be investigated
    assert (data_dir(cfg, "errors") / "rejected_interactions.csv").exists()


def test_clean_data_passes_without_faults(cfg):
    cfg["ingestion"]["inject_faults"] = False
    ingest.run(cfg)
    validate.run(cfg)
    report = read_json(data_dir(cfg, "reports") / "validation_report.json")
    assert report["tables"]["interactions"]["rows_rejected"] == 0
