"""Stage 2 - Data validation.

Every rule is *declared* (name, severity, description, action) and evaluated on the raw
CSV files (read as text so that wrong data types can be detected).

Severity
--------
* ``error``   -> the record is REJECTED and written to ``data/errors/rejected_*.csv``
* ``warning`` -> the record is kept, the bad value is set to NULL and imputed later
                 in the preparation stage

Outputs
-------
    data/processed/valid_{users,items,interactions}.parquet   typed, clean-enough records
    data/errors/rejected_{users,items,interactions}.csv       error dataset (+ failed_rules)
    data/errors/validation_issues.csv                         every issue, one row each
    reports/validation_report.{json,md}                       human-readable report
A *quality gate* stops the pipeline when too many interactions are rejected.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.common.config import config_hash, data_dir
from src.common.io import read_json, utc_now_iso, write_json
from src.common.log import get_logger

log = get_logger("validation")


class DataQualityError(RuntimeError):
    """Raised by the quality gate - downstream stages must not run."""


# Which injected fault (see ingestion/faults.py) each rule is expected to catch.
INJECTED_MAP = {
    ("interactions", "missing_user_id"): ["missing_user_id"],
    ("interactions", "missing_rating"): ["missing_rating"],
    ("interactions", "invalid_type_rating"): ["str_rating"],
    ("interactions", "rating_out_of_range"): ["bad_rating"],
    ("interactions", "invalid_type_timestamp"): ["bad_ts_str"],
    ("interactions", "timestamp_out_of_range"): ["bad_ts_neg", "bad_ts_future"],
    ("interactions", "duplicate_interaction"): ["duplicate_interaction"],
    ("interactions", "orphan_user_id"): ["orphan_user"],
    ("interactions", "orphan_item_id"): ["orphan_item"],
    ("users", "duplicate_user_id"): ["user_duplicate"],
    ("users", "missing_age"): ["user_missing_age"],
    ("users", "age_out_of_range"): ["user_age_out_of_range"],
    ("users", "invalid_type_age"): ["user_age_not_numeric"],
    ("users", "missing_gender"): ["user_missing_gender"],
    ("users", "invalid_gender"): ["user_invalid_gender"],
    ("users", "missing_occupation"): ["user_missing_occupation"],
    ("items", "duplicate_item_id"): ["item_duplicate"],
    ("items", "missing_title"): ["item_missing_title"],
    ("items", "invalid_release_date"): ["item_invalid_release_date"],
}


class RuleBook:
    """Collects rule results for one table."""

    def __init__(self, table: str, df: pd.DataFrame):
        self.table, self.df = table, df
        self.stats: list[dict] = []
        self.issues: list[pd.DataFrame] = []
        self.error_mask = pd.Series(False, index=df.index)
        self.warn_mask = pd.Series(False, index=df.index)
        self.reasons = pd.Series("", index=df.index, dtype=object)

    def check(self, rule: str, mask: pd.Series, severity: str, column: str, description: str, action: str):
        mask = mask.reindex(self.df.index).fillna(False).astype(bool)
        n = int(mask.sum())
        self.stats.append({
            "table": self.table, "rule": rule, "column": column, "severity": severity,
            "description": description, "action": action,
            "failed": n, "pct": round(100.0 * n / max(len(self.df), 1), 3),
        })
        if n == 0:
            return
        idx = mask[mask].index
        values = self.df.loc[idx, column]
        self.issues.append(pd.DataFrame({
            "table": self.table, "source_line": idx + 2,           # +2: header line + 1-based
            "rule": rule, "severity": severity, "column": column,
            "value": values.astype(object).where(values.notna(), "<NULL>").astype(str).values,
            "action": action,
        }))
        if severity == "error":
            self.error_mask |= mask
            self.reasons.loc[idx] = self.reasons.loc[idx].map(lambda s: f"{s};{rule}" if s else rule)
        else:
            self.warn_mask |= mask


def _read_raw(raw_dir, name: str) -> pd.DataFrame:
    # Everything is read as TEXT: type problems must be *detected*, not silently coerced by read_csv.
    return pd.read_csv(raw_dir / name, dtype=str, keep_default_na=True, na_values=["", "NA", "NaN", "nan", "None", "null"])


def _to_int(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (numeric values, mask of 'present but not an integer')."""
    num = pd.to_numeric(series, errors="coerce")
    present = series.notna()
    not_int = present & (num.isna() | (num != np.floor(num.fillna(0))))
    return num, not_int


def _validate_users(cfg, df):
    v = cfg["validation"]
    rb = RuleBook("users", df)
    uid, bad_id_type = _to_int(df["user_id"])
    rb.check("missing_user_id", df["user_id"].isna(), "error", "user_id", "user_id must not be NULL", "reject row")
    rb.check("invalid_type_user_id", bad_id_type, "error", "user_id", "user_id must be an integer", "reject row")
    ok_id = uid.notna() & ~bad_id_type
    rb.check("duplicate_user_id", ok_id & uid.duplicated(keep="first"), "error", "user_id",
             "user_id must be unique (first occurrence is kept)", "reject later duplicates")

    age = pd.to_numeric(df["age"], errors="coerce")
    rb.check("missing_age", df["age"].isna(), "warning", "age", "age should be present", "set NULL, impute (median)")
    rb.check("invalid_type_age", df["age"].notna() & age.isna(), "warning", "age", "age must be numeric",
             "set NULL, impute (median)")
    bad_range = age.notna() & ((age < v["age_min"]) | (age > v["age_max"]))
    rb.check("age_out_of_range", bad_range, "warning", "age",
             f"age must be within [{v['age_min']}, {v['age_max']}]", "set NULL, impute (median)")
    gender = df["gender"].str.strip().str.upper()
    rb.check("missing_gender", df["gender"].isna(), "warning", "gender", "gender should be present", "set to 'U' (unknown)")
    rb.check("invalid_gender", df["gender"].notna() & ~gender.isin(v["valid_genders"]), "warning", "gender",
             f"gender must be one of {v['valid_genders']}", "set to 'U' (unknown)")
    rb.check("missing_occupation", df["occupation"].isna(), "warning", "occupation",
             "occupation should be present", "set to 'unknown'")

    keep = ~rb.error_mask
    out = pd.DataFrame({
        "user_id": uid[keep].astype("int64"),
        "age": age[keep].where(~(bad_range[keep])),
        "gender": gender[keep],
        "occupation": df.loc[keep, "occupation"].str.strip(),
        "zip_code": df.loc[keep, "zip_code"].str.strip(),
    }).reset_index(drop=True)
    return rb, out


def _validate_items(cfg, df):
    rb = RuleBook("items", df)
    iid, bad_id_type = _to_int(df["item_id"])
    rb.check("missing_item_id", df["item_id"].isna(), "error", "item_id", "item_id must not be NULL", "reject row")
    rb.check("invalid_type_item_id", bad_id_type, "error", "item_id", "item_id must be an integer", "reject row")
    ok_id = iid.notna() & ~bad_id_type
    rb.check("duplicate_item_id", ok_id & iid.duplicated(keep="first"), "error", "item_id",
             "item_id must be unique (first occurrence is kept)", "reject later duplicates")
    rb.check("missing_title", df["title"].isna(), "warning", "title", "title should be present", "set to 'Unknown title'")
    rel = pd.to_datetime(df["release_date"], format="%d-%b-%Y", errors="coerce")
    rb.check("invalid_release_date", df["release_date"].notna() & rel.isna(), "warning", "release_date",
             "release_date must look like 01-Jan-1995", "set NULL, impute (median year)")
    rb.check("missing_genres", df["genres"].isna(), "warning", "genres", "genres should be present", "set to 'unknown'")

    keep = ~rb.error_mask
    out = pd.DataFrame({
        "item_id": iid[keep].astype("int64"),
        "title": df.loc[keep, "title"],
        "release_date": rel[keep],
        "genres": df.loc[keep, "genres"],
    }).reset_index(drop=True)
    return rb, out


def _validate_interactions(cfg, df, valid_user_ids: set, valid_item_ids: set):
    v = cfg["validation"]
    rb = RuleBook("interactions", df)

    uid, bad_u = _to_int(df["user_id"])
    iid, bad_i = _to_int(df["item_id"])
    rating = pd.to_numeric(df["rating"], errors="coerce")
    ts = pd.to_numeric(df["timestamp"], errors="coerce")

    # --- completeness
    for col in ("user_id", "item_id", "rating", "timestamp"):
        rb.check(f"missing_{col}", df[col].isna(), "error", col, f"{col} must not be NULL", "reject row")
    # --- data types
    rb.check("invalid_type_user_id", bad_u, "error", "user_id", "user_id must be an integer", "reject row")
    rb.check("invalid_type_item_id", bad_i, "error", "item_id", "item_id must be an integer", "reject row")
    rb.check("invalid_type_rating", df["rating"].notna() & rating.isna(), "error", "rating",
             "rating must be numeric", "reject row")
    rb.check("invalid_type_timestamp", df["timestamp"].notna() & ts.isna(), "error", "timestamp",
             "timestamp must be numeric (unix seconds)", "reject row")
    # --- domain rules
    rb.check("rating_out_of_range", rating.notna() & ((rating < v["rating_min"]) | (rating > v["rating_max"])),
             "error", "rating", f"rating must be within [{v['rating_min']}, {v['rating_max']}]", "reject row")
    lo = pd.Timestamp(v["min_timestamp"], tz="UTC").timestamp()
    hi = pd.Timestamp.now(tz="UTC").timestamp() if str(v["max_timestamp"]) == "now" \
        else pd.Timestamp(v["max_timestamp"], tz="UTC").timestamp()
    rb.check("timestamp_out_of_range", ts.notna() & ((ts < lo) | (ts > hi)), "error", "timestamp",
             f"timestamp must be between {v['min_timestamp']} and {v['max_timestamp']}", "reject row")
    # --- uniqueness
    key_ok = uid.notna() & iid.notna() & ts.notna() & ~bad_u & ~bad_i
    key = pd.DataFrame({"u": uid, "i": iid, "t": ts})[key_ok]
    dup = key.duplicated(keep="first")
    dup_mask = pd.Series(False, index=df.index)
    dup_mask.loc[dup[dup].index] = True
    rb.check("duplicate_interaction", dup_mask, "error", "user_id",
             "(user_id, item_id, timestamp) must be unique (first occurrence is kept)", "reject later duplicates")
    # --- referential integrity
    rb.check("orphan_user_id", uid.notna() & ~bad_u & ~uid.isin(valid_user_ids), "error", "user_id",
             "user_id must exist in the (valid) users table", "reject row")
    rb.check("orphan_item_id", iid.notna() & ~bad_i & ~iid.isin(valid_item_ids), "error", "item_id",
             "item_id must exist in the (valid) items table", "reject row")

    keep = ~rb.error_mask
    out = pd.DataFrame({
        "user_id": uid[keep].astype("int64"),
        "item_id": iid[keep].astype("int64"),
        "rating": rating[keep].astype("float64"),
        "timestamp": ts[keep].astype("int64"),
    }).reset_index(drop=True)
    return rb, out


def _write_rejected(raw_df: pd.DataFrame, rb: RuleBook, path):
    bad = raw_df[rb.error_mask].copy()
    bad.insert(0, "source_line", bad.index + 2)
    bad["failed_rules"] = rb.reasons[rb.error_mask]
    bad.to_csv(path, index=False)


def _markdown(report: dict) -> str:
    lines = ["# Data validation report", "",
             f"*Generated: {report['generated_at']}  |  config hash: `{report['config_hash']}`  |  data source: {report['source']}*", ""]
    lines += ["## 1. Summary per table", "",
              "| Table | Rows in | Valid rows | Rejected rows | Reject rate | Rows with warnings |",
              "|---|---:|---:|---:|---:|---:|"]
    for t, s in report["tables"].items():
        lines.append(f"| {t} | {s['rows_in']:,} | {s['rows_valid']:,} | {s['rows_rejected']:,} | "
                     f"{s['reject_rate_pct']}% | {s['rows_with_warnings']:,} |")
    g = report["gate"]
    lines += ["", f"**Quality gate:** interactions reject rate {g['interaction_reject_rate_pct']}% "
              f"(threshold {g['max_error_rate_pct']}%) -> **{'PASSED' if g['passed'] else 'FAILED'}**", ""]
    lines += ["## 2. Rule results", "",
              "| Table | Rule | Column | Severity | Failed | % | Injected (demo) | Action |",
              "|---|---|---|---|---:|---:|---:|---|"]
    for r in report["rules"]:
        inj = r.get("injected")
        lines.append(f"| {r['table']} | `{r['rule']}` | {r['column']} | {r['severity']} | {r['failed']:,} | {r['pct']} | "
                     f"{'' if inj is None else inj} | {r['action']} |")
    lines += ["", "*'Injected (demo)' = number of corrupt records deliberately inserted by the ingestion stage "
              "(`ingestion.inject_faults`). Detected >= injected shows that the rule works.*", "",
              "## 3. Where the bad records went", "",
              "* `data/errors/rejected_users.csv`, `rejected_items.csv`, `rejected_interactions.csv` - rejected rows with the failed rule(s)",
              "* `data/errors/validation_issues.csv` - one line per issue (errors and warnings)",
              "* `data/processed/valid_*.parquet` - records that passed validation", ""]
    if report.get("examples"):
        lines += ["## 4. Example rejected records", ""]
        for rule, rows in report["examples"].items():
            lines.append(f"**{rule}**")
            lines.append("")
            lines.append("| source_line | column | bad value |")
            lines.append("|---:|---|---|")
            for r in rows:
                lines.append(f"| {r['source_line']} | {r['column']} | `{r['value']}` |")
            lines.append("")
    return "\n".join(lines)


def run(cfg: dict) -> dict:
    t0 = time.time()
    raw, proc, err, rep = (data_dir(cfg, k) for k in ("raw", "processed", "errors", "reports"))
    users_raw, items_raw, inter_raw = (_read_raw(raw, f) for f in ("users.csv", "items.csv", "interactions.csv"))

    rb_u, users = _validate_users(cfg, users_raw)
    rb_i, items = _validate_items(cfg, items_raw)
    rb_x, inter = _validate_interactions(cfg, inter_raw, set(users["user_id"]), set(items["item_id"]))

    users.to_parquet(proc / "valid_users.parquet", index=False)
    items.to_parquet(proc / "valid_items.parquet", index=False)
    inter.to_parquet(proc / "valid_interactions.parquet", index=False)
    _write_rejected(users_raw, rb_u, err / "rejected_users.csv")
    _write_rejected(items_raw, rb_i, err / "rejected_items.csv")
    _write_rejected(inter_raw, rb_x, err / "rejected_interactions.csv")
    issues = pd.concat([x for rb in (rb_u, rb_i, rb_x) for x in rb.issues] or [pd.DataFrame()], ignore_index=True)
    issues.to_csv(err / "validation_issues.csv", index=False)

    manifest_path = raw / "ingestion_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    injected = manifest.get("faults_injected", {})

    rules = []
    for rb in (rb_u, rb_i, rb_x):
        for s in rb.stats:
            keys = INJECTED_MAP.get((s["table"], s["rule"]))
            s = dict(s)
            s["injected"] = sum(injected.get(k, 0) for k in keys) if (keys and injected) else None
            rules.append(s)

    tables = {}
    for name, rb, raw_df, clean in (("users", rb_u, users_raw, users), ("items", rb_i, items_raw, items),
                                    ("interactions", rb_x, inter_raw, inter)):
        rejected = int(rb.error_mask.sum())
        tables[name] = {"rows_in": len(raw_df), "rows_valid": len(clean), "rows_rejected": rejected,
                        "reject_rate_pct": round(100 * rejected / max(len(raw_df), 1), 3),
                        "rows_with_warnings": int((rb.warn_mask & ~rb.error_mask).sum())}

    reject_rate = tables["interactions"]["reject_rate_pct"] / 100
    passed = reject_rate <= cfg["validation"]["max_error_rate"] and len(inter) >= cfg["validation"]["min_valid_rows"]

    examples = {}
    if len(issues):
        for rule, grp in issues[issues["severity"] == "error"].groupby("rule"):
            examples[rule] = grp.head(3)[["source_line", "column", "value"]].to_dict("records")

    report = {
        "generated_at": utc_now_iso(), "config_hash": config_hash(cfg),
        "source": manifest.get("source", "unknown"),
        "tables": tables, "rules": rules, "examples": examples,
        "gate": {"interaction_reject_rate_pct": round(100 * reject_rate, 3),
                 "max_error_rate_pct": 100 * cfg["validation"]["max_error_rate"],
                 "min_valid_rows": cfg["validation"]["min_valid_rows"], "passed": bool(passed)},
    }
    write_json(rep / "validation_report.json", report)
    (rep / "validation_report.md").write_text(_markdown(report), encoding="utf-8")
    log.info("Validation: interactions %d in -> %d valid, %d rejected (%.2f%%)",
             tables["interactions"]["rows_in"], tables["interactions"]["rows_valid"],
             tables["interactions"]["rows_rejected"], 100 * reject_rate)

    if not passed:
        raise DataQualityError(
            f"Quality gate FAILED: {100 * reject_rate:.2f}% of interactions rejected "
            f"(max allowed {100 * cfg['validation']['max_error_rate']:.1f}%), valid rows = {len(inter)}. "
            f"See reports/validation_report.md and data/errors/.")
    return {"stage": "validation", "finished_at": utc_now_iso(), "tables": tables,
            "gate_passed": True, "seconds": round(time.time() - t0, 2)}
