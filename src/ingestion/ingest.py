"""Stage 1 - Data ingestion.

Produces three raw CSV files with a fixed schema, regardless of the data source:

    data/raw/users.csv         user_id, age, gender, occupation, zip_code
    data/raw/items.csv         item_id, title, release_date, genres  ("a|b|c")
    data/raw/interactions.csv  user_id, item_id, rating, timestamp   (unix seconds)
"""
from __future__ import annotations

import time

from src.common.config import config_hash, data_dir
from src.common.io import file_sha256, utc_now_iso, write_json
from src.common.log import get_logger
from src.ingestion.faults import inject_faults
from src.ingestion.movielens import ensure_movielens, parse_movielens_100k
from src.ingestion.synthetic import generate_synthetic

log = get_logger("ingestion")


def run(cfg: dict) -> dict:
    t0 = time.time()
    icfg = cfg["ingestion"]
    raw_dir = data_dir(cfg, "raw")
    seed = cfg["project"]["seed"]
    source = icfg["source"]

    users = items = inter = None
    if source == "movielens":
        try:
            folder = ensure_movielens(raw_dir, icfg["movielens_url"])
            users, items, inter = parse_movielens_100k(folder)
            log.info("Loaded MovieLens-100K from %s", folder)
        except Exception as exc:  # network down, SSL problem, corrupt zip ...
            if not icfg.get("fallback_to_synthetic", False):
                raise
            log.warning("MovieLens load failed (%s: %s) -> falling back to SYNTHETIC data. "
                        "Put ml-100k.zip into data/raw/ to use the real dataset.", type(exc).__name__, exc)
            source = "synthetic (fallback)"
    if users is None:
        s = icfg["synthetic"]
        users, items, inter = generate_synthetic(s["n_users"], s["n_items"], s["n_interactions"], seed)
        source = source if source.startswith("synthetic") else "synthetic"
        log.info("Generated synthetic data: %d users, %d items, %d interactions",
                 len(users), len(items), len(inter))

    injected: dict = {}
    if icfg.get("inject_faults", False):
        users, items, inter, injected = inject_faults(users, items, inter, seed, icfg.get("fault_rate", 0.002))
        log.info("Injected demo faults into raw data: %s", injected)

    users.to_csv(raw_dir / "users.csv", index=False)
    items.to_csv(raw_dir / "items.csv", index=False)
    inter.to_csv(raw_dir / "interactions.csv", index=False)

    summary = {
        "stage": "ingestion",
        "finished_at": utc_now_iso(),
        "source": source,
        "config_hash": config_hash(cfg),
        "rows": {"users": len(users), "items": len(items), "interactions": len(inter)},
        "sha256": {f: file_sha256(raw_dir / f) for f in ("users.csv", "items.csv", "interactions.csv")},
        "faults_injected": injected,
        "seconds": round(time.time() - t0, 2),
    }
    write_json(raw_dir / "ingestion_manifest.json", summary)
    log.info("Ingestion done: %s", summary["rows"])
    return summary
