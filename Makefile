# Convenience shortcuts (macOS / Linux). Run `make help`.
PY := .venv/bin/python
export PREFECT_SERVER_ANALYTICS_ENABLED := false
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help setup run flow test mlflow-ui prefect-server clean

help:
	@echo "make setup          create .venv and install requirements"
	@echo "make run            run the whole pipeline with the plain-Python runner"
	@echo "make flow           run the whole pipeline as a Prefect flow"
	@echo "make test           run the unit tests"
	@echo "make mlflow-ui      open the MLflow UI on http://127.0.0.1:5001"
	@echo "make prefect-server start the Prefect UI on http://127.0.0.1:4200"
	@echo "make clean          delete generated data, models, MLflow DB (keeps code + reports)"

setup:
	uv venv --python 3.12 .venv
	uv pip install -r requirements.txt

run:
	$(PY) run_pipeline.py

flow:
	$(PY) dags/recommendation_flow.py

test:
	$(PY) -m pytest -q

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

prefect-server:
	.venv/bin/prefect server start

clean:
	rm -rf data/raw/* data/processed/* data/features/* data/errors/* models/*.joblib mlruns mlflow.db logs/*
	touch data/raw/.gitkeep data/processed/.gitkeep data/features/.gitkeep data/errors/.gitkeep models/.gitkeep logs/.gitkeep
