# End-to-End Data Management Pipeline for a Recommendation System
<em class="sub">Data Management for Machine Learning - Assignment 2 &nbsp;|&nbsp; Student: <b>[YOUR NAME / ID]</b> &nbsp;|&nbsp; Repository: <b>[GITHUB LINK]</b></em>

## 1. Problem and data
The goal is a reproducible pipeline that ingests raw user, item and interaction data, validates and prepares it, manages features,
trains and evaluates recommenders, and runs automatically. Data source: **movielens** - 946 users, 1,685 items and 100,400 raw
interaction rows (rating 1-5 with a unix timestamp; items carry genres). Because the source data are (almost) perfectly clean, a small, seeded amount of corruption was injected into the raw files on purpose (flag `ingestion.inject_faults`) so that every rule has something to catch; 2,006 rows were rejected in total.

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
The rule book has 28 rules over the three tables. Records with an *error* (missing key, wrong type, rating outside
1-5, impossible timestamp, duplicate event, orphan user/item) are **rejected**
into `data/errors/`; records with a *warning* (missing/invalid age, gender, title, release date) are kept and repaired later.
Of 100,400 interaction rows, **98,400 were valid and 2,000 rejected (1.992%)**.
A quality gate stops the pipeline if more than 5% of interactions are rejected
(status: **passed**). Full details: `reports/validation_report.md`.

## 4. Preparation and transformation
Cleaning imputes missing age (median) and release year, standardises gender/occupation/genre spelling and keeps the *latest* rating
when a user rated an item twice. The **time-based split** (per-user: each user's most recent
20% of interactions form the test set) gives 78,916 train and 19,200 test
interactions; an assertion proves that no training event is later than a test event of the same user. Test rows of unseen
users/items (84) are dropped (cold start is out of scope). Derived datasets, all built from train only:
user activity summary, item popularity, a 943 x 1,682 sparse user-item matrix (95.0% sparse), genre multi-hot matrix.

## 5. Feature management
The catalog (`reports/feature_catalog.md`) documents **29 features** across three entities - user (activity count, average
rating, positive rate, recency, preferred category, genre-affinity vector, demographics), item (popularity count/rank/score, average and
Bayesian-damped rating, age, genres) and user-item pair (genre affinity, preferred-category match, rating gap). Features are computed
as of the last training timestamp (point-in-time correct) by `FeatureStore`; the hybrid model calls the same `genre_affinity_matrix`
method at training and at prediction time, which avoids training/serving skew.

## 6. Models and experiment tracking
* **Popularity baseline** - recommends the most-interacted items the user has not seen (also the cold-start fallback).
* **SVD (personalised)** - two heads: truncated SVD of the user-item matrix (32 factors, a.k.a. PureSVD) for
  *ranking*, and a biased matrix factorisation fitted by ALS (r = mu + b_user + b_item + p.q, 16 factors) for *rating prediction*.
* **Hybrid** - per-user standardised SVD score blended with the genre-affinity score from the feature store.

Each model is one MLflow run storing parameters (factors, seed, split, feature version), metrics (fit time, explained variance, all test
metrics) and artifacts (model file, config, feature catalog, results).

## 7. Evaluation results
Protocol: relevant = test rating >= 4; 903 users evaluated; top-10 lists exclude items already seen in train.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity (baseline) | 0.0683 | 0.0721 | 0.0871 | 0.040 | 0.750 | 1.088 | 0.873 |
| SVD matrix factorisation | 0.1208 | 0.1575 | 0.1659 | 0.265 | 0.715 | 0.991 | 0.782 |
| Hybrid (SVD + genre) | 0.1194 | 0.1567 | 0.1661 | 0.269 | 0.670 | 0.991 | 0.782 |

* The SVD model improves on the popularity baseline: NDCG@10 0.1659 vs 0.0871 (+91%), Precision@10 0.1208 vs 0.0683.
* Coverage of the catalog: popularity 4.0%, SVD 26.5%, hybrid 26.9%; list diversity: 0.750 / 0.715 / 0.670.
* The hybrid model changes NDCG@10 by +0.1% relative to plain SVD (content weight 0.2); the genre signal adds little on top of the collaborative signal.
* For explicit rating prediction the SVD rating head reaches RMSE 0.991 / MAE 0.782, against 1.088 / 0.873 for the damped item-mean baseline.

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
