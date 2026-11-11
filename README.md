# End-to-End Data Management Pipeline for a Recommendation System

*Demonstrated on [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) — works with any dataset matching the required schema.*

An automated, reproducible pipeline that turns raw user / item / interaction data into evaluated
recommendation models:

```
Ingestion -> Validation -> Preparation -> Transformation -> Feature Management -> Training -> Evaluation -> Reporting
                                  (orchestrated by Prefect  /  run_pipeline.py)
```

* **Dataset:** [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) (943 users, 1,682 movies, 100,000 ratings).
  If the download is impossible the pipeline falls back to a documented synthetic generator.
* **Models:** popularity baseline, SVD matrix factorisation (personalised), hybrid (SVD + genre affinity).
* **Tracking:** MLflow. **Orchestration:** Prefect (plus a plain-Python runner). **Storage:** CSV + Parquet + sparse `.npz`.

## Architecture

![architecture](docs/architecture.png)

```mermaid
flowchart LR
    S[(MovieLens-100K<br/>or synthetic)] --> A[1 Ingestion]
    A -->|raw CSV| B[2 Validation]
    B -->|valid parquet| C[3 Preparation<br/>time-based split]
    B -.->|rejected rows| E[(data/errors)]
    C --> D[4 Transformation]
    D --> F[5 Feature store<br/>+ catalog]
    F --> G[6 Training]
    G --> H[7 Evaluation]
    G <-->|params, metrics, artifacts| M[(MLflow)]
    H --> M
    H --> R[8 Reports]
    O{{Prefect flow / run_pipeline.py}} -.-> A & B & C & D & F & G & H & R
```

## Quick start (macOS / Linux)

> **Servers overview** — three optional UI servers exist in this project. None are required to run the pipeline itself.
>
> | Server | Address | Purpose | When to start |
> |---|---|---|---|
> | MLflow UI | `http://127.0.0.1:5001` | Browse experiment runs, compare model metrics, download artifacts | After the pipeline runs, to inspect results |
> | Prefect UI | `http://127.0.0.1:4200` | View DAG timeline, task logs, and schedules | When using the Prefect orchestrator |
> | Prefect ephemeral | `http://127.0.0.1:8890` (auto) | Temporary Prefect API — starts automatically when no server is running | Never start manually; avoid by setting `PREFECT_API_URL` |

### Step 1 — Set up the environment

No servers needed. This only installs dependencies.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Shortcut: `make setup`

> **Offline / SSL problems?** Download <https://files.grouplens.org/datasets/movielens/ml-100k.zip> manually,
> put the zip into `data/raw/` and run again - the download is skipped.
> To use synthetic data instead: `python run_pipeline.py --set ingestion.source=synthetic`.

### Step 2 — Run the pipeline

No servers needed. All results are written to disk (`data/`, `models/`, `reports/`, `mlflow.db`).

There are two ways to run the pipeline — both execute the same 8 stages with the same logic:

| | `run_pipeline.py` | `dags/recommendation_flow.py` |
|---|---|---|
| **What it is** | Plain Python script | Prefect flow |
| **Extra dependencies** | None | Prefect |
| **Retries on failure** | No | Yes — ingestion retries on network error |
| **Scheduling** | No | Yes — `--serve` reruns daily at 02:00 |
| **Observability** | Terminal logs only | DAG timeline + task logs in Prefect UI |
| **Best for** | Quick local runs, CI, debugging | Scheduled runs with monitoring |

```bash
# Option A: plain Python runner — quickest, no extra setup
python run_pipeline.py

# Option B: Prefect flow — use when you want scheduling, retries, or the UI
python dags/recommendation_flow.py
```

Shortcut: `make run` (Option A) · `make flow` (Option B)

**What happens during a run:**

| Step | Source file | What it does |
|---|---|---|
| 1. Load config | `src/common/config.py` | Reads `configs/config.yaml`, applies any `--set` overrides |
| 2. Ingest | `src/ingestion/ingest.py` | Downloads MovieLens-100K (or synthetic data), injects demo faults, writes raw CSVs to `data/raw/` |
| 3. Validate | `src/validation/validate.py` | Runs the rule book; rejects error rows to `data/errors/`; aborts if >5% of interactions are rejected |
| 4. Prepare | `src/preparation/prepare.py` | Cleans, imputes, de-duplicates; performs time-based train/test split; writes Parquet to `data/processed/` |
| 5. Transform | `src/transformation/transform.py` | Builds user activity summary, item popularity, sparse user-item matrix |
| 6. Features | `src/features/build_features.py` | Computes 29 features (user, item, pair); writes to `data/features/`; generates `reports/feature_catalog.md` |
| 7. Train | `src/training/train.py` | Trains 3 models (popularity, SVD, hybrid); logs params/metrics/artifacts to MLflow (`mlflow.db`) |
| 8. Evaluate | `src/evaluation/evaluate.py` | Computes Precision/Recall/NDCG/Coverage/Diversity/RMSE/MAE; writes `reports/model_comparison.*` |
| 9. Report | `src/reporting/make_reports.py` | Generates architecture diagram, final report draft, updates the README results block |
| 10. Manifest | `src/pipeline.py` | Saves `reports/run_manifest.json` (config hash, git commit, package versions, checksums) |

If any stage fails, the run stops immediately and the manifest is written with the failure details.

### Step 3 — Inspect model metrics (MLflow UI)

Start the MLflow UI **after** the pipeline has run. It reads from `mlflow.db` — no data is lost if the server is not running during execution.

```bash
# Port 5000 is taken by AirPlay on macOS, so use 5001
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open `http://127.0.0.1:5001`, click **Experiments** in the left sidebar, and select `recommendation-pipeline`:

![MLflow experiments](docs/screenshots/ml-flow-experiments.png)

Click **Runs** to see all 3 models with their metrics side by side:

![MLflow runs](docs/screenshots/ml-flow-run.png)

### Step 4 — Run with the orchestrator (Prefect)

The Prefect UI (port 4200) must be running **before** the flow if you want schedules to persist. Without it, Prefect starts a temporary ephemeral server that cannot save schedules across restarts.

```bash
# Option A: run once and exit — no Prefect server needed
python dags/recommendation_flow.py

# Option B: run on a daily schedule (persistent) — start the Prefect server first
prefect server start                                # terminal 1  ->  http://127.0.0.1:4200
export PREFECT_API_URL=http://127.0.0.1:4200/api    # terminal 2 (same folder, venv active)
python dags/recommendation_flow.py --serve          # reruns every day at 02:00
```

> **Ephemeral server warning** — if you run `--serve` without setting `PREFECT_API_URL`, Prefect
> auto-starts a temporary server (e.g. `http://127.0.0.1:8890`). The flow runs, but the schedule
> is lost when the process exits. Always start `prefect server start` first for scheduled runs.

After the run, open `http://127.0.0.1:4200` → **Runs** → click the flow run to see the stage timeline:

![Prefect DAG timeline](docs/screenshots/perfect_dag.png)

Each box is one pipeline stage. The arrows show the dependency chain. The x-axis is wall-clock time.

### Step 5 — Run the unit tests

No servers needed.

```bash
python -m pytest -q
```

Shortcut: `make test`

### Useful options

```bash
# Skip early stages; rerun only train, evaluate, and report (data already processed)
python run_pipeline.py --stages train evaluate report

# Override config values without editing config.yaml:
#   content_weight=0.4  -> genre affinity influence in the hybrid model
#   primary_k=5         -> recommendation list length (default 10)
python run_pipeline.py --set models.hybrid.content_weight=0.4 evaluation.primary_k=5

# Disable fault injection; run validation on the clean, unmodified MovieLens data
python run_pipeline.py --set ingestion.inject_faults=false

# Change train/test split from per-user (each user's last 20%) to a single global cut-off date
python run_pipeline.py --set preparation.split_strategy=global
```

### Validations

The validation stage (`src/validation/validate.py`) runs a declarative rule book across all three tables. Rules have two severities:

- **Error** — row is rejected and written to `data/errors/rejected_*.csv`
- **Warning** — row is kept, bad value is nulled and imputed later in preparation

**Interactions**

| Rule | Severity |
|---|---|
| Missing `user_id`, `item_id`, `rating`, or `timestamp` | Error |
| Non-integer `user_id` or `item_id` | Error |
| `rating` not numeric | Error |
| `timestamp` not numeric | Error |
| Rating outside 1–5 | Error |
| Timestamp before 1990-01-01 or in the future | Error |
| Duplicate `(user_id, item_id, timestamp)` | Error |
| `user_id` not in users table (orphan) | Error |
| `item_id` not in items table (orphan) | Error |

**Users**

| Rule | Severity |
|---|---|
| Missing or non-integer `user_id` | Error |
| Duplicate `user_id` | Error |
| Missing or non-numeric `age` | Warning |
| Age outside 5–100 | Warning |
| Missing or invalid `gender` (must be M or F) | Warning |
| Missing `occupation` | Warning |

**Items**

| Rule | Severity |
|---|---|
| Missing or non-integer `item_id` | Error |
| Duplicate `item_id` | Error |
| Missing `title` | Warning |
| Invalid `release_date` format | Warning |
| Missing `genres` | Warning |

**Quality gate** — the pipeline aborts if more than 5% of interaction rows are rejected. The threshold is configurable via `validation.max_error_rate` in `configs/config.yaml`. Full results are written to `reports/validation_report.md`.

### Using a different dataset

The pipeline supports three data sources. No code changes are needed for the built-in sources.

| Source | How to use |
|---|---|
| MovieLens-100K | Default — downloaded automatically on first run |
| Synthetic | `python run_pipeline.py --set ingestion.source=synthetic` |
| Your own CSV files | Drop 3 files into `data/raw/` matching the schema below, then run normally |

Your CSV files must use this exact schema:

| File | Required columns |
|---|---|
| `data/raw/users.csv` | `user_id, age, gender, occupation, zip_code` |
| `data/raw/items.csv` | `item_id, title, release_date, genres` (genres as `"action\|comedy"`) |
| `data/raw/interactions.csv` | `user_id, item_id, rating, timestamp` (rating 1–5, timestamp in unix seconds) |

**Limitations without code changes:** rating scale is fixed at 1–5, genres must be pipe-separated, and the 3-table structure (users / items / interactions) is required.

### Data storage

The pipeline is entirely file-based — no database is required to run it.

**What is created after each step:**

| After step | Files written | Location |
|---|---|---|
| Ingest | `users.csv`, `items.csv`, `interactions.csv` | `data/raw/` |
| Validate | `rejected_*.csv` (error rows) | `data/errors/` |
| Prepare + Transform | Parquet files, `interaction_matrix.npz` | `data/processed/` |
| Features | Parquet feature tables | `data/features/` |
| Train | `.joblib` model files | `models/` |
| Train (MLflow) | Params, metrics, artifacts | `mlflow.db` (SQLite) |
| Report | Markdown, CSV, PNG, HTML | `reports/` |

**Moving to a database (PostgreSQL / SQLite):**

The pipeline currently writes CSVs after ingestion and Parquet after preparation. To load data into a database instead, the following changes are needed:

| File to change | What to change |
|---|---|
| `src/ingestion/ingest.py` | Replace `df.to_csv()` with `df.to_sql()` using a SQLAlchemy engine |
| `src/validation/validate.py` | Read from DB with `pd.read_sql()` instead of `pd.read_csv()` |
| `src/preparation/prepare.py` | Write processed tables to DB instead of Parquet |
| `configs/config.yaml` | Add a `database.url` key (e.g. `postgresql://user:pass@localhost/rec`) |

The rest of the pipeline (features, training, evaluation) operates on in-memory DataFrames and does not need changes — only the read/write boundaries in ingestion, validation, and preparation need updating.

### Modifying features

All feature changes go through a single file: **`src/features/definitions.py`** — the single source of truth for the feature catalog, feature store, and models.

**Add a new feature:**
1. Add a `FeatureDef(...)` entry to the `FEATURES` list in `src/features/definitions.py`
2. Implement the computation in `src/features/store.py`
3. Register it in `src/features/build_features.py` so it is computed and saved during the pipeline run

**Remove a feature:**
1. Check the `used_by` field on the definition — if a model uses it (e.g. `genre_affinity` is used by Hybrid), update `src/training/models.py` first
2. Delete the `FeatureDef` entry from `definitions.py`
3. Remove the corresponding method from `src/features/store.py`

**Modify an existing feature:**
1. Update the `logic` field in the `FeatureDef` in `definitions.py` (keeps docs in sync)
2. Change the computation in `src/features/store.py`

**Change a threshold (no code change needed):**

Edit `configs/config.yaml`:

| Key | Default | Effect |
|---|---|---|
| `features.positive_threshold` | `4` | Minimum rating to count as "liked" |
| `features.bayes_prior` | `10` | Damping strength for Bayesian item average |

After any feature change, rerun from the features stage:

```bash
python run_pipeline.py --stages features train evaluate report
```

### Models

Three models are trained in `src/training/train.py` and tracked in MLflow.

**1. Popularity (baseline)**
- Recommends the most-interacted items the user has not seen
- No personalisation — same ranked list for all users
- Used as the cold-start fallback for users/items with no history
- Rating prediction uses Bayesian-damped item average

**2. SVD — matrix factorisation (personalised)**
- Two heads trained independently:
  - **Ranking head** — PureSVD (TruncatedSVD on the user-item matrix, 32 factors) for top-K recommendations
  - **Rating head** — Biased ALS matrix factorisation (`r = μ + b_user + b_item + p·q`, 16 factors) for explicit rating prediction
- `scikit-learn` only — no compiler needed on macOS

**3. Hybrid (SVD + genre affinity)**
- Blends the SVD ranking score with the `genre_affinity` feature from the feature store
- Formula: `score = (1 - content_weight) × svd_score + content_weight × genre_affinity_score`
- Default `content_weight = 0.2` — override with `--set models.hybrid.content_weight=0.4`
- Uses the same `genre_affinity_matrix` at training and prediction time — no training/serving skew

| Model | Source file | Config key |
|---|---|---|
| Popularity | `src/training/models.py` | `models.popularity` |
| SVD | `src/training/models.py` | `models.svd` |
| Hybrid | `src/training/models.py` | `models.hybrid` |

All models log params, metrics, and artifacts to MLflow (`mlflow.db`) on every run.

Trained model files are saved to the `models/` folder:

```
models/
├── popularity.joblib
├── svd.joblib
└── hybrid.joblib
```

The output path is configurable via `paths.models` in `configs/config.yaml`.

### Testing the models manually

Use `tests/test_models.py` to get recommendations from all three models for any user:

```bash
# Test all 3 models with default users, top-5 recommendations
python tests/test_models.py

# Specific user IDs and top-10
python tests/test_models.py --users 1 42 100 --k 10

# One model only
python tests/test_models.py --model svd --users 259 --k 5
```

Example output:

```
User IDs : [259, 851]
Top-K    : 3

==================================================
Model: popularity
==================================================
  user 259 (predicted rating: 2.77)
    1. Star Wars (1977)
    2. Fargo (1996)
    3. Contact (1997)

==================================================
Model: svd
==================================================
  user 259 (predicted rating: 3.11)
    1. Contact (1997)
    2. Star Wars (1977)
    3. Pulp Fiction (1994)

==================================================
Model: hybrid
==================================================
  user 259 (predicted rating: 3.11)
    1. Contact (1997)
    2. Star Wars (1977)
    3. Pulp Fiction (1994)
```

> The pipeline must have run at least once before testing — models are loaded from `models/*.joblib`.

## Project structure

```
recommendation-pipeline/
├── configs/config.yaml        all settings (paths, thresholds, hyper-parameters, seeds)
├── dags/recommendation_flow.py   Prefect flow = the orchestration DAG
├── run_pipeline.py            plain-Python runner (same stage functions as the flow)
├── src/
│   ├── common/                config loading, logging, I/O helpers
│   ├── ingestion/             MovieLens loader, synthetic generator, fault injector (demo)
│   ├── validation/            declarative rule book, error dataset, quality gate
│   ├── preparation/           cleaning, imputation, de-duplication, TIME-BASED split
│   ├── transformation/        normalisation, encodings, user/item summaries, sparse matrix
│   ├── features/              feature registry/catalog + feature store (train & predict)
│   ├── training/              models (popularity, SVD, hybrid) + MLflow tracking
│   ├── evaluation/            Precision/Recall/NDCG@K, coverage, diversity, RMSE, MAE
│   ├── reporting/             architecture diagram, report draft, README results block
│   └── pipeline.py            stage registry + run manifest
├── data/{raw,processed,features,errors}/
├── models/   reports/   docs/   tests/
```


## Results (latest run)

> These stats are written automatically by the `report` stage at the end of every pipeline run.
> To refresh them, run `python run_pipeline.py` (full run) or `python run_pipeline.py --stages report` (report only).
> The source data is `reports/model_comparison.json` and `reports/validation_report.json`.

<!-- RESULTS:START -->
_Auto-generated by the pipeline (stage `report`) on 2026-11-11T12:37:09Z_

**Evaluation (K=10, time-based test set)**

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0683 | 0.0721 | 0.0871 | 0.040 | 0.750 | 1.088 | 0.873 |
| svd | 0.1208 | 0.1575 | 0.1659 | 0.265 | 0.715 | 0.991 | 0.782 |
| hybrid | 0.1194 | 0.1567 | 0.1661 | 0.269 | 0.670 | 0.991 | 0.782 |

**Validation:** 100,400 interaction rows in -> 98,400 valid, 2,000 rejected (1.992%) - see `reports/validation_report.md`.
<!-- RESULTS:END -->

