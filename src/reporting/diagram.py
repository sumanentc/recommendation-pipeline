"""Draws the architecture diagram (docs/architecture.png) with matplotlib - no extra tools needed."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

STAGES = [
    ("1  Ingestion", "CSV / public data\nusers, items,\ninteractions\n(+ demo faults)", "#dbeafe"),
    ("2  Validation", "nulls · types ·\nranges · duplicates\nreferential integrity\nQUALITY GATE", "#fee2e2"),
    ("3  Preparation", "clean · impute\nde-duplicate\nTIME-BASED\ntrain / test split", "#fef3c7"),
    ("4  Transform", "normalise · encode\nuser activity,\nitem popularity,\nuser-item matrix", "#fef3c7"),
    ("5  Features", "feature store\nuser / item / pair\nfeatures +\nfeature catalog", "#dcfce7"),
    ("6  Training", "Popularity (base)\nSVD (MF)\nHybrid\n+ MLflow tracking", "#ede9fe"),
    ("7  Evaluation", "Precision · Recall\nNDCG @K\nCoverage · Diversity\nRMSE · MAE", "#ede9fe"),
    ("8  Reporting", "final report\nmodel comparison\narchitecture diagram\nREADME results", "#e0f2fe"),
]

STORAGE = [
    (0, 1, "data/raw", "users.csv\nitems.csv\ninteractions.csv"),
    (1, 1, "data/errors", "rejected_*.csv\nvalidation_issues"),
    (2, 2, "data/processed", "clean + train/test parquet\nmatrix.npz, summaries"),
    (4, 1, "data/features", "user / item /\ninteraction features"),
    (5, 1, "models/ · MLflow", "joblib models\nparams · metrics"),
    (7, 1, "reports/", "model_comparison\nfeature_catalog\nfinal report"),
]


def draw_architecture(path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    ax.text(8, 8.1, "End-to-end recommendation data pipeline - architecture", ha="center", va="center",
            fontsize=16, fontweight="bold")

    # orchestration band
    ax.add_patch(FancyBboxPatch((0.4, 6.75), 15.6, 0.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#f3e8ff", ec="#7e22ce", lw=1.5))
    ax.text(8, 7.2, "ORCHESTRATION  -  Prefect flow (dags/recommendation_flow.py)  or  run_pipeline.py",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#581c87")
    ax.text(8, 6.93, "task dependencies · retries · quality gate stops downstream tasks · run manifest (config hash, checksums)",
            ha="center", va="center", fontsize=9.5, color="#581c87")

    # data source
    ax.add_patch(FancyBboxPatch((0.4, 4.35), 1.65, 1.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#f1f5f9", ec="#475569", lw=1.2))
    ax.text(1.2, 5.2, "Data source\n\nMovieLens-100K\n(or synthetic)", ha="center", va="center", fontsize=9)

    x0, w, gap = 2.3, 1.6, 0.1
    centers = []
    for i, (title, body, color) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 4.35), w, 1.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=color, ec="#334155", lw=1.3))
        ax.text(x + w / 2, 5.78, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, 5.05, body, ha="center", va="center", fontsize=8.3, linespacing=1.35)
        centers.append(x + w / 2)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 5.2), (x + w + gap, 5.2), arrowstyle="-|>", mutation_scale=14, color="#334155"))
    ax.add_patch(FancyArrowPatch((2.0, 5.2), (x0, 5.2), arrowstyle="-|>", mutation_scale=14, color="#334155"))

    # storage layer
    ax.text(1.2, 2.78, "STORAGE\n(files on disk)", ha="center", va="center", fontsize=9, color="#475569", style="italic")
    for start, span, title, body in STORAGE:
        x = x0 + start * (w + gap)
        width = span * w + (span - 1) * gap
        ax.add_patch(FancyBboxPatch((x, 2.15), width, 1.25, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="#f8fafc", ec="#94a3b8", lw=1.1))
        ax.text(x + width / 2, 3.12, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + width / 2, 2.62, body, ha="center", va="center", fontsize=8.3, linespacing=1.3, color="#334155")
        for s in range(span):
            cx = x + w / 2 + s * (w + gap)
            ax.add_patch(FancyArrowPatch((cx, 4.33), (cx, 3.42), arrowstyle="<|-|>", mutation_scale=9,
                                         color="#94a3b8", lw=1, linestyle="--"))

    # cross-cutting
    ax.add_patch(FancyBboxPatch((0.4, 0.45), 15.6, 1.35, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#ecfdf5", ec="#047857", lw=1.3))
    ax.text(8, 1.5, "CROSS-CUTTING CONCERNS", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#065f46")
    ax.text(8, 0.98,
            "Reproducibility: fixed seeds · single YAML config · config hash + data checksums in run manifest · pytest unit tests\n"
            "Observability: logs/pipeline.log · validation report · MLflow UI (params, metrics, artifacts per model)\n"
            "Leakage control: features + matrices computed from the TRAIN split only · temporal-leakage assertion in the split",
            ha="center", va="center", fontsize=9, linespacing=1.5, color="#064e3b")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
