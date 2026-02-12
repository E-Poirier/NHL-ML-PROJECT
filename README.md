# NHL Hockey Performance Prediction System

Binary classification: predict whether a player will record ≥1 point (goal or assist) in their next game, **given they play**.

See [PLAN.md](PLAN.md) for the full end-to-end ML pipeline specification.

## Project Structure

```
├── config/          # Configuration (paths, API, training, retrain)
├── data/            # raw/, processed/, features/
├── src/             # Core modules (ingestion, validation, features, train, etc.)
├── api/             # FastAPI prediction endpoint
├── monitoring/      # Drift detection, daily evaluation
├── models/          # Model artifacts (v1/, v2/, ...)
├── scripts/         # run_ingestion.sh, run_retrain.sh, demo_predict.py
├── tests/           # Unit and contract tests
└── notebooks/       # EDA, SHAP analysis
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Run

| Step | Command |
|------|---------|
| Ingest data | `./scripts/run_ingestion.sh` or `python -m src.data_ingestion` |
| Validate | `./scripts/run_validation.sh` or `python -m src.validation` |
| Features | `./scripts/run_feature_engineering.sh` or `python -m src.feature_engineering` |
| Train | `python -m src.train` |
| Start API | `uvicorn api.main:app --reload` |
| Demo predict | `python scripts/demo_predict.py` |

## Config

Edit `config/config.yaml` for paths, seasons, model version, and retrain thresholds. Use environment variables for secrets.

## Model Registry

`model_registry.json` tracks current production model and version history. See PLAN.md Section 9.
