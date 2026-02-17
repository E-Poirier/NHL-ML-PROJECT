# NHL Hockey Performance Prediction System

Binary classification: predict whether a player will record ≥1 point (goal or assist) in their next game, **given they play**.

See [PLAN.md](PLAN.md) for the full end-to-end ML pipeline specification.

---

## Problem Definition

### Target

**y = 1:** Player played in the game and recorded ≥1 point.  
**y = 0:** Player played in the game and recorded 0 points.

The model answers: *"Given this player plays in this game, what is the probability they record ≥1 point?"*

### DNP Handling

- **Excluded:** Any player–game where the player did not play (DNP: scratched, injured, no TOI). These rows are not in the training or evaluation dataset.
- **"Played" definition:** `time_on_ice > 0` or presence in the game's skater box score.
- **Prediction time:** The API returns P(≥1 point | player plays). Callers combine with their own "will they play?" belief if needed.

### Cold Start

Players with fewer than `min_games_history` (default: 5) games played before a given game are excluded from features. The API returns 404 with `insufficient_history: true` for player–games not in the feature table.

---

## Data

### Source

- **NHL API** via `nhlpy` (api-web.nhle.com).
- Box scores, schedule, and roster data; rate limits and retries configured in `config/config.yaml`.

### Layout

| Directory        | Contents                                      |
|------------------|-----------------------------------------------|
| `data/raw/YYYY_MM_DD/` | Raw JSON box scores, `game_ids.json`          |
| `data/processed/`     | Validated parquet: `player_games_YYYY_MM_DD.parquet` |
| `data/features/`      | Feature parquet: `features_YYYY_MM_DD.parquet` |

---

## Train/Val/Test Splits

- **Train:** Earlier seasons (e.g. 2018–19 through 2021–22) — model training.
- **Validation:** Next season (e.g. 2022–23) — hyperparameter tuning and threshold selection.
- **Test:** Following season (e.g. 2023–24) — final evaluation only; never used for tuning.

Splits are time-based (by season). Class imbalance is handled with `scale_pos_weight` (XGBoost).

---

## Project Structure

```
├── config/          # config.yaml (paths, API, training, retrain, monitoring)
├── data/            # raw/, processed/, features/
├── src/             # ingestion, validation, feature_engineering, train, retrain
├── api/             # FastAPI prediction endpoint
├── monitoring/      # Drift detection, daily evaluation
├── models/          # Model artifacts (v1/, v2/, ...)
├── scripts/         # run_ingestion.sh, run_daily.sh, run_retrain.sh, run_monitoring.sh, crontab.example
├── tests/           # Unit, contract, and integration tests
├── dashboard/       # Streamlit UI
└── notebooks/       # EDA, SHAP analysis
```

---

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

| Step        | Command                                                       |
|------------|---------------------------------------------------------------|
| Ingest data | `./scripts/run_ingestion.sh` or `python -m src.data_ingestion` |
| Validate   | `./scripts/run_validation.sh` or `python -m src.validation`    |
| Features   | `./scripts/run_feature_engineering.sh` or `python -m src.feature_engineering` |
| Train      | `./scripts/run_train.sh` or `python -m src.train`              |
| Start API  | `uvicorn api.main:app --reload` (from project root)            |
| Demo predict | `python scripts/demo_predict.py`                             |
| Dashboard | `./scripts/run_dashboard.sh` or `streamlit run dashboard/app.py` |
| Monitoring | `./scripts/run_monitoring.sh` or `python -m monitoring.monitor` |
| Retrain    | `./scripts/run_retrain.sh` or `python -m src.retrain`          |

**Retrain options:** `--skip-ingestion` to reuse existing raw data; `--force-promotion` to promote regardless of criterion; `--check-only` to check if retrain is triggered.

**Scheduling (cron):** `scripts/run_daily.sh` runs ingestion → validation → features → monitoring. Copy the two cron lines from `scripts/crontab.example`, replace `/path/to/project` with your project path, run `mkdir -p logs`, then paste into `crontab -e` and save. Daily job at 6 AM; monthly retrain on the 1st at 9 AM.

---

## Notebooks

- **SHAP analysis** (`notebooks/shap_analysis.ipynb`): feature importance and explainability. Run the first code cell to `%pip install shap`, restart the kernel, then run all. Use the project `.venv` as the Jupyter kernel so `xgboost` is available: from project root, `python -m ipykernel install --user --name=nhl-ml --display-name "Python (NHL ML .venv)"`, then in Jupyter choose **Kernel → Change kernel → "Python (NHL ML .venv)"**.
- **EDA** (`notebooks/eda.ipynb`): exploratory data analysis.

---

## Docker

Build and run the API in a container. **Requires** `models/` and `data/features/` (run the pipeline first).

```bash
docker compose up --build
```

API at http://localhost:8000. Override port in `docker-compose.yml` if needed. To point the dashboard at a different API host: `API_URL=http://host:8000 streamlit run dashboard/app.py`.

---

## Config

Edit `config/config.yaml` for:

- **paths:** raw_data, processed_data, features, models, monitoring
- **training:** train_seasons, val_season, test_season, random_seed
- **features:** min_games_history, rolling windows
- **api:** model_version (fallback if no registry), cold_start_min_games
- **monitoring:** recent_days, eval_days
- **retrain:** performance_floor_roc_auc, drift_tolerance_zscore, test_degradation_tolerance_pct, schedule (monthly/weekly/on_demand)

Use environment variables for secrets; never commit API keys.

---

## Model Registry

`model_registry.json` tracks:

- **current_production:** Active model version (e.g. `v2`)
- **versions:** History with val/test ROC-AUC, F1, trained_at

The API reads the registry and serves the current production model.

### Promotion

A new model is promoted only if:

1. Validation ROC-AUC ≥ current production, and  
2. Test ROC-AUC does not degrade beyond `test_degradation_tolerance_pct` (default 5%).

Retrain triggers: schedule (monthly/weekly), performance drop (from monitoring), or feature/prediction drift.

---

## Tests

```bash
pytest tests/ -v
```

- **test_validation.py** — Schema, types, duplicates, DNP exclusion
- **test_feature_engineering.py** — Point-in-time guarantee, cold start
- **test_api.py** — Contract and cold-start behavior
- **test_retrain.py** — Registry, check_retrain_needed
- **test_monitor.py** — Monitoring helpers and outputs
- **test_integration.py** — Minimal train → save → load → predict flow (requires feature data)
