"""Stage 7 - Model evaluation on the held-out (later in time) test interactions.

Protocol
--------
* relevant items of a user = test items the user rated >= relevance_threshold
* users evaluated         = users with at least one relevant test item
* each model recommends top-K items per user, items already seen in TRAIN are excluded
* ranking metrics: Precision@K, Recall@K, NDCG@K, Coverage@K, Diversity@K   (K in k_values)
* rating metrics : RMSE, MAE on all test ratings
Results are written to reports/ and logged into the MLflow run of each model.
"""
from __future__ import annotations

import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.config import data_dir  # noqa: E402
from src.common.io import read_json, utc_now_iso, write_json  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.evaluation.metrics import (catalog_coverage, intra_list_diversity, mae,  # noqa: E402
                                    precision_recall_ndcg_at_k, rmse)
from src.training.tracking import setup_mlflow  # noqa: E402

log = get_logger("evaluation")

DISPLAY = {"popularity": "Popularity (baseline)", "svd": "SVD matrix factorisation", "hybrid": "Hybrid (SVD + genre)"}


def _md_table(df: pd.DataFrame, k: int) -> str:
    cols = [("model", "Model"), (f"precision_at_{k}", f"Precision@{k}"), (f"recall_at_{k}", f"Recall@{k}"),
            (f"ndcg_at_{k}", f"NDCG@{k}"), (f"coverage_at_{k}", f"Coverage@{k}"),
            (f"diversity_at_{k}", f"Diversity@{k}"), ("rmse", "RMSE"), ("mae", "MAE")]
    lines = ["| " + " | ".join(label for _, label in cols) + " |", "|---|" + "---:|" * (len(cols) - 1)]
    for _, r in df.iterrows():
        cells = [DISPLAY.get(r["model"], r["model"])] + [f"{r[c]:.4f}" for c, _ in cols[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plot(df: pd.DataFrame, k: int, path) -> None:
    metrics = [(f"precision_at_{k}", f"Precision@{k}"), (f"recall_at_{k}", f"Recall@{k}"),
               (f"ndcg_at_{k}", f"NDCG@{k}"), (f"coverage_at_{k}", f"Coverage@{k}"), (f"diversity_at_{k}", f"Diversity@{k}")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(df)
    x = np.arange(len(metrics))
    palette = ["#9aa5b1", "#1f77b4", "#e07b39"]
    for j, (_, r) in enumerate(df.iterrows()):
        ax.bar(x + j * width, [r[m] for m, _ in metrics], width, label=DISPLAY.get(r["model"], r["model"]),
               color=palette[j % len(palette)])
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("score")
    ax.set_title(f"Model comparison on the time-based test set (K={k})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sample_recommendations(models: dict, train, test, items, relevant_by_user, eval_users, thr, path, n_users=3, k=5):
    title = items.set_index("item_id")["title"]
    genres = items.set_index("item_id")["genres"]
    lines = ["# Sample recommendations", "",
             "Qualitative check for a few users: what they liked in the past, what each model recommends, "
             "and which of the recommended items they actually liked later (marked with a check).", ""]
    picked = [u for u in eval_users if len(relevant_by_user[u]) >= 3][:n_users]
    for u in picked:
        liked = (train[(train["user_id"] == u) & (train["rating"] >= thr)].sort_values("rating", ascending=False).head(5))
        lines += [f"## User {u}", "", "**Liked in the past (train):** " +
                  "; ".join(f"{title[i]} [{genres[i]}]" for i in liked["item_id"]), ""]
        for name, model in models.items():
            rec = model.recommend([u], k)[0]
            marks = "; ".join(f"{title[i]}{' ✓' if i in relevant_by_user[u] else ''}" for i in rec)
            lines.append(f"* **{DISPLAY.get(name, name)}:** {marks}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict) -> dict:
    t0 = time.time()
    proc, models_dir, rep = data_dir(cfg, "processed"), data_dir(cfg, "models"), data_dir(cfg, "reports")
    ecfg = cfg["evaluation"]
    ks, primary_k, thr = ecfg["k_values"], ecfg["primary_k"], ecfg["relevance_threshold"]

    train, test = pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "test.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet")
    genre_df = pd.read_parquet(proc / "item_genre_matrix.parquet")
    item_ids = genre_df["item_id"].to_numpy()
    G = genre_df.drop(columns="item_id").to_numpy(dtype="float64")

    rel = test[test["rating"] >= thr].groupby("user_id")["item_id"].apply(set)
    eval_users = np.sort(rel.index.to_numpy())
    relevant_by_user = rel.to_dict()
    relevant = [relevant_by_user[u] for u in eval_users]
    log.info("Evaluating on %d users (%d relevant test interactions)", len(eval_users), sum(map(len, relevant)))

    manifest = read_json(models_dir / "training_manifest.json")
    models = {name: joblib.load(cfg["_root"] + "/" + info["path"]) for name, info in manifest["models"].items()}

    kmax = max(ks)
    rows = []
    for name, model in models.items():
        recs = model.recommend(eval_users, kmax)
        row = {"model": name, "n_users_evaluated": len(eval_users)}
        for k in ks:
            p, r, n = precision_recall_ndcg_at_k(recs, relevant, k)
            row[f"precision_at_{k}"], row[f"recall_at_{k}"], row[f"ndcg_at_{k}"] = p.mean(), r.mean(), n.mean()
            row[f"coverage_at_{k}"] = catalog_coverage(recs, len(item_ids), k)
            row[f"diversity_at_{k}"] = intra_list_diversity(recs, item_ids, G, k)
        pred = model.predict_rating(test["user_id"].to_numpy(), test["item_id"].to_numpy())
        row["rmse"], row["mae"] = rmse(test["rating"], pred), mae(test["rating"], pred)
        rows.append(row)
        log.info("%-10s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.3f div=%.3f RMSE=%.3f MAE=%.3f", name, primary_k,
                 row[f"precision_at_{primary_k}"], primary_k, row[f"recall_at_{primary_k}"], primary_k,
                 row[f"ndcg_at_{primary_k}"], row[f"coverage_at_{primary_k}"], row[f"diversity_at_{primary_k}"],
                 row["rmse"], row["mae"])

    res = pd.DataFrame(rows)
    res.to_csv(rep / "model_comparison.csv", index=False)
    md = ["# Model evaluation results", "",
          f"Time-based test set: {len(test):,} interactions, {len(eval_users):,} evaluated users "
          f"(relevant = test rating >= {thr}). Items seen in training are never recommended.", ""]
    for k in sorted(ks, reverse=True):
        md += [f"## Top-{k}", "", _md_table(res, k), ""]
    md += ["*Coverage = share of the catalog recommended to at least one user. Diversity = mean pairwise genre distance "
           "inside a recommendation list. RMSE/MAE = rating-prediction error on all test ratings.*", ""]
    (rep / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    write_json(rep / "model_comparison.json", {"primary_k": primary_k, "k_values": ks, "results": rows})
    _plot(res, primary_k, rep / "model_comparison.png")
    _sample_recommendations(models, train, test, items, relevant_by_user, eval_users, thr,
                            rep / "sample_recommendations.md")

    # ---- log into the MLflow run of each model
    setup_mlflow(cfg)
    for r in rows:
        run_id = manifest["models"][r["model"]]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({k: float(v) for k, v in r.items() if k not in ("model",)})
            mlflow.set_tag("evaluated", "true")
            for f in ("model_comparison.csv", "model_comparison.md", "model_comparison.png", "sample_recommendations.md"):
                mlflow.log_artifact(str(rep / f), artifact_path="evaluation")
    return {"stage": "evaluation", "finished_at": utc_now_iso(), "n_users_evaluated": int(len(eval_users)),
            "results": rows, "seconds": round(time.time() - t0, 2)}
