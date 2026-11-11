"""Generates an auto-filled DRAFT of the 2-3 page final report (Markdown + printable HTML).

All numbers come from the artifacts of the latest pipeline run, so the report always matches the results.
Edit `reports/final_report_draft.md` (add your name, refine the wording), then re-render with

    python -m src.reporting.final_report reports/final_report_draft.md

and print the HTML to PDF from the browser (Cmd+P -> Save as PDF).
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import markdown
import scipy.sparse as sp

from src.common.config import data_dir, load_config
from src.common.io import read_json
from src.features.definitions import FEATURES

CSS = """
@page { size: A4; margin: 13mm 14mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #1f2933; max-width: 860px; margin: 0 auto; padding: 12px; }
h1 { font-size: 17pt; margin: 0 0 2px 0; color: #0f172a; }
h2 { font-size: 11.5pt; margin: 13px 0 4px 0; border-bottom: 1px solid #cbd5e1; padding-bottom: 2px; color: #0f172a; }
p, ul, ol { margin: 4px 0; }
ul, ol { padding-left: 18px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8.7pt; }
th, td { border: 1px solid #cbd5e1; padding: 3px 6px; text-align: left; }
th { background: #f1f5f9; }
code { background: #f1f5f9; padding: 0 3px; border-radius: 3px; font-size: 8.6pt; }
img { max-width: 100%; }
em.sub { color: #64748b; }
"""


def _pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b if b else float("nan")


def build_markdown(cfg: dict) -> str:
    raw, proc, feat, rep, models = (data_dir(cfg, k) for k in ("raw", "processed", "features", "reports", "models"))
    ing = read_json(raw / "ingestion_manifest.json")
    val = read_json(rep / "validation_report.json")
    split = read_json(proc / "split_info.json")
    res = read_json(rep / "model_comparison.json")
    fmeta = read_json(feat / "feature_metadata.json")
    train_manifest = read_json(models / "training_manifest.json")["models"]
    R = sp.load_npz(proc / "interaction_matrix.npz")
    sparsity = 100 * (1 - R.nnz / (R.shape[0] * R.shape[1]))

    k = res["primary_k"]
    r = {x["model"]: x for x in res["results"]}
    pop, svd, hyb = r["popularity"], r["svd"], r["hybrid"]
    n_x, n_u, n_i = ing["rows"]["interactions"], ing["rows"]["users"], ing["rows"]["items"]
    vx = val["tables"]["interactions"]
    faults = bool(ing.get("faults_injected"))
    src = ing["source"]
    nm = lambda m, key: m[f"{key}_at_{k}"]  # noqa: E731

    # --- results table
    table = ["| Model | Precision@%d | Recall@%d | NDCG@%d | Coverage@%d | Diversity@%d | RMSE | MAE |" % ((k,) * 5),
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"popularity": "Popularity (baseline)", "svd": "SVD matrix factorisation", "hybrid": "Hybrid (SVD + genre)"}
    for name in ("popularity", "svd", "hybrid"):
        m = r[name]
        table.append(f"| {labels[name]} | {nm(m, 'precision'):.4f} | {nm(m, 'recall'):.4f} | {nm(m, 'ndcg'):.4f} | "
                     f"{nm(m, 'coverage'):.3f} | {nm(m, 'diversity'):.3f} | {m['rmse']:.3f} | {m['mae']:.3f} |")

    better = "improves on" if nm(svd, "ndcg") > nm(pop, "ndcg") else "does not improve on"
    hy = nm(hyb, "ndcg") - nm(svd, "ndcg")
    hy_txt = (f"The hybrid model changes NDCG@{k} by {_pct(nm(hyb, 'ndcg'), nm(svd, 'ndcg')):+.1f}% relative to plain SVD "
              f"(content weight {cfg['models']['hybrid']['content_weight']}); " +
              ("blending in genre affinity therefore helps slightly." if hy > 0.0005 else
               "the genre signal adds little on top of the collaborative signal." if abs(hy) <= 0.0005 else
               "the genre signal does not help on this data."))
    rmse_txt = (f"For explicit rating prediction the SVD rating head reaches RMSE {svd['rmse']:.3f} / MAE {svd['mae']:.3f}, "
                f"against {pop['rmse']:.3f} / {pop['mae']:.3f} for the damped item-mean baseline.")

    fault_txt = ""
    if faults:
        detected = sum(t["rows_rejected"] for t in val["tables"].values())
        fault_txt = (f" Because the source data are (almost) perfectly clean, a small, seeded amount of corruption was injected into the raw "
                     f"files on purpose (flag `ingestion.inject_faults`) so that every rule has something to catch; "
                     f"{detected:,} rows were rejected in total.")

    md = f"""# End-to-End Data Management Pipeline for a Recommendation System
<em class="sub">Data Management for Machine Learning - Assignment 2 &nbsp;|&nbsp; Student: <b>[YOUR NAME / ID]</b> &nbsp;|&nbsp; Repository: <b>[GITHUB LINK]</b></em>

## 1. Problem and data
The goal is a reproducible pipeline that ingests raw user, item and interaction data, validates and prepares it, manages features,
trains and evaluates recommenders, and runs automatically. Data source: **{src}** - {n_u:,} users, {n_i:,} items and {n_x:,} raw
interaction rows (rating 1-5 with a unix timestamp; items carry genres).{fault_txt}

## 2. Architecture and tools
![architecture](ARCH_IMAGE_PLACEHOLDER)

| Stage | Implementation | Why this choice |
|---|---|---|
| Ingestion | Python + pandas; MovieLens download or synthetic generator | one fixed raw schema regardless of source |
| Validation | declarative rule book in pandas (nulls, types, ranges, duplicates, referential integrity) | every rule is named, counted and testable; rejected rows go to an error dataset |
| Preparation / transformation | pandas, scipy.sparse, Parquet | columnar storage, typed, fast to reload |
| Feature management | custom feature store + feature catalog | training and prediction share one implementation; lighter than Feast for a project of this size |
| Training / tracking | scikit-learn (TruncatedSVD), MLflow (SQLite backend) | pure-Python, installs everywhere (no compiler needed, unlike LightFM / implicit) |
| Orchestration | Prefect flow, plus `run_pipeline.py` fallback runner | explicit task dependencies, retries, UI; Airflow needs a heavy scheduler/DB set-up |

## 3. Data validation
The rule book has {len(val['rules'])} rules over the three tables. Records with an *error* (missing key, wrong type, rating outside
{cfg['validation']['rating_min']}-{cfg['validation']['rating_max']}, impossible timestamp, duplicate event, orphan user/item) are **rejected**
into `data/errors/`; records with a *warning* (missing/invalid age, gender, title, release date) are kept and repaired later.
Of {vx['rows_in']:,} interaction rows, **{vx['rows_valid']:,} were valid and {vx['rows_rejected']:,} rejected ({vx['reject_rate_pct']}%)**.
A quality gate stops the pipeline if more than {100 * cfg['validation']['max_error_rate']:.0f}% of interactions are rejected
(status: **{'passed' if val['gate']['passed'] else 'failed'}**). Full details: `reports/validation_report.md`.

## 4. Preparation and transformation
Cleaning imputes missing age (median) and release year, standardises gender/occupation/genre spelling and keeps the *latest* rating
when a user rated an item twice. The **time-based split** ({split['strategy'].replace('_', '-')}: each user's most recent
{int(100 * split['test_fraction'])}% of interactions form the test set) gives {split['n_train']:,} train and {split['n_test']:,} test
interactions; an assertion proves that no training event is later than a test event of the same user. Test rows of unseen
users/items ({split['n_test_dropped_cold_start']:,}) are dropped (cold start is out of scope). Derived datasets, all built from train only:
user activity summary, item popularity, a {R.shape[0]:,} x {R.shape[1]:,} sparse user-item matrix ({sparsity:.1f}% sparse), genre multi-hot matrix.

## 5. Feature management
The catalog (`reports/feature_catalog.md`) documents **{len(FEATURES)} features** across three entities - user (activity count, average
rating, positive rate, recency, preferred category, genre-affinity vector, demographics), item (popularity count/rank/score, average and
Bayesian-damped rating, age, genres) and user-item pair (genre affinity, preferred-category match, rating gap). Features are computed
as of the last training timestamp (point-in-time correct) by `FeatureStore`; the hybrid model calls the same `genre_affinity_matrix`
method at training and at prediction time, which avoids training/serving skew.

## 6. Models and experiment tracking
* **Popularity baseline** - recommends the most-interacted items the user has not seen (also the cold-start fallback).
* **SVD (personalised)** - two heads: truncated SVD of the user-item matrix ({cfg['models']['svd']['n_factors']} factors, a.k.a. PureSVD) for
  *ranking*, and a biased matrix factorisation fitted by ALS (r = mu + b_user + b_item + p.q, {cfg['models']['svd']['rating_factors']} factors) for *rating prediction*.
* **Hybrid** - per-user standardised SVD score blended with the genre-affinity score from the feature store.

Each model is one MLflow run storing parameters (factors, seed, split, feature version), metrics (fit time, explained variance, all test
metrics) and artifacts (model file, config, feature catalog, results).

## 7. Evaluation results
Protocol: relevant = test rating >= {cfg['evaluation']['relevance_threshold']}; {res['results'][0]['n_users_evaluated']:,} users evaluated; top-{k} lists exclude items already seen in train.

{chr(10).join(table)}

* The SVD model {better} the popularity baseline: NDCG@{k} {nm(svd, 'ndcg'):.4f} vs {nm(pop, 'ndcg'):.4f} ({_pct(nm(svd, 'ndcg'), nm(pop, 'ndcg')):+.0f}%), Precision@{k} {nm(svd, 'precision'):.4f} vs {nm(pop, 'precision'):.4f}.
* Coverage of the catalog: popularity {nm(pop, 'coverage'):.1%}, SVD {nm(svd, 'coverage'):.1%}, hybrid {nm(hyb, 'coverage'):.1%}; list diversity: {nm(pop, 'diversity'):.3f} / {nm(svd, 'diversity'):.3f} / {nm(hyb, 'diversity'):.3f}.
* {hy_txt}
* {rmse_txt}

## 8. Orchestration and reproducibility
The Prefect flow (`dags/recommendation_flow.py`) has one task per stage; each task receives its upstream task's result, so the DAG is
ingest -> validate -> prepare -> transform -> features -> train -> evaluate -> report, and a failed quality gate stops everything downstream.
Ingestion retries on network errors. Reproducibility comes from the YAML config (hashed into every run), fixed seeds, a run manifest
(`reports/run_manifest.json`: config hash, package versions, git commit, file checksums) and a pytest suite.

## 9. Deliverables index

| Deliverable | Location |
|---|---|
| Source code | `src/` |
| Setup and execution instructions | `README.md` |
| Dependencies | `requirements.txt` |
| Architecture diagram | `docs/architecture.png` |
| Data validation report | `reports/validation_report.md`, `data/errors/*.csv`, `logs/pipeline.log` |
| Feature catalog | `reports/feature_catalog.md` / `.csv` |
| Model evaluation results | `reports/model_comparison.md` / `.csv` / `.png`, `reports/sample_recommendations.md` |
| Orchestration workflow | `dags/recommendation_flow.py` |
| Screenshots of successful run | `docs/screenshots/` |
| Final report | `reports/final_report_draft.html` (print to PDF via Cmd+P) |
| Run manifest (reproducibility) | `reports/run_manifest.json` |

## 10. Limitations and future work
* Offline evaluation on a small dataset; no online/A-B test; hyper-parameters (factors, content weight) were not tuned.
* The per-user chronological split lets other users' later events appear in training; a single global cut-off date avoids this but leaves fewer evaluable users.
* Cold-start users/items only receive popularity recommendations; the ranking head treats unobserved ratings as zeros, whereas implicit-feedback models (ALS with confidence weights, BPR) model them explicitly.
* The custom feature store is file-based (no low-latency online serving); Feast or a database-backed store would be the production step.
* Validation is a hand-written rule book; Great Expectations / Pandera would add reusable expectations and data docs.
* Next steps: scheduled retraining with drift monitoring, hyper-parameter search tracked in MLflow, and model registry promotion.
"""
    return md


def render_html(md_text: str, out_path: Path, arch_image: Path | None) -> None:
    html_body = markdown.markdown(md_text, extensions=["tables"])
    if arch_image and Path(arch_image).exists():
        b64 = base64.b64encode(Path(arch_image).read_bytes()).decode()
        html_body = html_body.replace("ARCH_IMAGE_PLACEHOLDER", f"data:image/png;base64,{b64}")
    out_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Final report</title>"
                        f"<style>{CSS}</style></head><body>{html_body}</body></html>", encoding="utf-8")


def run(cfg: dict) -> dict:
    rep, docs = data_dir(cfg, "reports"), data_dir(cfg, "docs")
    md = build_markdown(cfg)
    (rep / "final_report_draft.md").write_text(md, encoding="utf-8")
    render_html(md, rep / "final_report_draft.html", docs / "architecture.png")
    return {"report_md": "reports/final_report_draft.md", "report_html": "reports/final_report_draft.html"}


if __name__ == "__main__":  # re-render an edited markdown file:  python -m src.reporting.final_report <file.md>
    cfg_ = load_config()
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else data_dir(cfg_, "reports") / "final_report_draft.md"
    render_html(target.read_text(encoding="utf-8"), target.with_suffix(".html"), data_dir(cfg_, "docs") / "architecture.png")
    print(f"Rendered {target.with_suffix('.html')} - open it in Chrome and use Cmd+P -> Save as PDF")
