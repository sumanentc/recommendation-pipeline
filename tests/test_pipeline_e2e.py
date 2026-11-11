import pandas as pd

from src.common.config import data_dir
from src.common.io import read_json
from src.pipeline import run_all


def test_full_pipeline_and_reproducibility(cfg):
    run_all(cfg)
    rep = data_dir(cfg, "reports")
    for f in ("validation_report.md", "feature_catalog.md", "model_comparison.csv", "model_comparison.png",
              "final_report_draft.html", "run_manifest.json", "sample_recommendations.md"):
        assert (rep / f).exists(), f
    assert (data_dir(cfg, "docs") / "architecture.png").exists()

    res = pd.read_csv(rep / "model_comparison.csv").set_index("model")
    assert set(res.index) == {"popularity", "svd", "hybrid"}
    for col in ("precision_at_10", "recall_at_10", "ndcg_at_10", "coverage_at_10", "diversity_at_10", "rmse", "mae"):
        assert col in res.columns and res[col].notna().all()
    # on structured synthetic data a personalised model must beat the popularity baseline
    assert res.loc["svd", "ndcg_at_10"] > res.loc["popularity", "ndcg_at_10"]
    assert res.loc["svd", "rmse"] < res.loc["popularity", "rmse"]
    assert read_json(rep / "run_manifest.json")["status"] == "success"

    # MLflow: one run per model with params, metrics and artifacts
    import mlflow
    from src.training.tracking import setup_mlflow
    setup_mlflow(cfg)
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment_name"]])
    assert len(runs) == 3
    assert runs["metrics.ndcg_at_10"].notna().all() and runs["params.model"].notna().all()

    # reproducibility: same config + seed -> identical results
    first = (rep / "model_comparison.csv").read_text()
    run_all(cfg)
    assert (rep / "model_comparison.csv").read_text() == first
