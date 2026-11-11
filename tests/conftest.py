import pytest

from src.common.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """A small synthetic configuration that writes everything into a temporary folder."""
    cfg = load_config(overrides=[
        "ingestion.source=synthetic",
        "ingestion.synthetic.n_users=120",
        "ingestion.synthetic.n_items=100",
        "ingestion.synthetic.n_interactions=4000",
        "validation.min_valid_rows=500",
        "mlflow.experiment_name=pytest-experiment",
    ])
    cfg["_root"] = str(tmp_path)          # all relative paths (data/, models/, reports/ ...) now live in tmp_path
    return cfg
