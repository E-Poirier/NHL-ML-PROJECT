# NHL Hockey Performance Prediction System

## Detailed End-to-End ML Pipeline Plan (Revised)

---

## 1. Define the Prediction Problem

### 1.1 Target

Binary classification: Predict whether a player will record at least 1 point (goal or assist) in their next game.

**Condition:** Prediction is conditional on the player playing in that game. The model answers: "Given this player plays in this game, what is the probability they record ≥1 point?"

### 1.2 Label Definition

- **y = 1:** Player played in the game and recorded ≥1 point.
- **y = 0:** Player played in the game and recorded 0 points.
- **Excluded:** Any player–game where the player did not play (DNP: scratched, injured, no TOI). These rows are not included in the training or evaluation dataset.

### 1.3 DNP Handling

- **Training data:** Only include player–games where the player actually played. Use a "played" definition: e.g. `time_on_ice > 0` or presence in the game's skater box score (or equivalent from your data source).
- **Prediction time:** Either (a) only return predictions for players expected to play (e.g. from lineup/roster), or (b) document that the API returns "P(≥1 point | player plays)" and let callers combine with their own "will they play?" belief.
- **Cold start:** Define policy for players with little or no history (see Section 5.5).

---

## 2. System Architecture Overview

```
NHL API / Data Source
        ↓
Raw Data Storage (timestamped)
        ↓
Data Validation (schema, duplicates, DNP filter)
        ↓
Processed Data Storage
        ↓
Feature Engineering Pipeline (point-in-time)
        ↓
Dataset Construction (train/val/test, stratified by time)
        ↓
Model Training (baseline + XGBoost, class weights, threshold tuning)
        ↓
Evaluation + Metrics (incl. calibration, test set)
        ↓
Model Versioning + Artifact Storage + Registry
        ↓
Feature Serving (batch job → feature store / table)
        ↓
Prediction API (FastAPI)
        ↓
Monitoring + Drift Detection
        ↓
Daily Performance Evaluation Job (predictions vs outcomes)
        ↓
Retraining Trigger + Promotion Criterion
```

---

## 3. Automated Stat Ingestion

### 3.1 Data Source

- **Prefer:** NHL public API (e.g. api-web.nhle.com) or a documented sports data API.
- Document which API is used, rate limits, and any fallback (e.g. caching, retries).
- Avoid manual CSV downloads; all ingestion is scripted.

### 3.2 Implementation

**File:** `src/data_ingestion.py`

**Responsibilities:**
- Pull game-level and player-level stats (box scores, TOI, goals, assists, shots, etc.).
- Pull schedule data (dates, home/away, opponent).
- Pull roster/participation data to support "played" vs DNP.
- Store raw API responses (JSON) with no transformation.
- Timestamp each run (e.g. `data/raw/YYYY_MM_DD/` or per-run subfolder).
- **Retries and backoff:** Implement retry logic (e.g. exponential backoff) and respect rate limits to handle transient failures.

**Raw storage path:** `data/raw/YYYY_MM_DD/` (or equivalent with run id).

**Key fields to capture (at least):**
- `game_id`, `player_id`, `team_id`, `date`, `opponent`, `home_away`
- `goals`, `assists`, `points`, `shots`, `time_on_ice`, `plus_minus`
- Roster/participation or TOI so "played" can be defined.

### 3.3 Scheduling

- Cron or scheduled Python script (e.g. daily post-games).
- Optional: GitHub Actions for a public repo (e.g. daily trigger).
- Config (e.g. `config/ingestion.yaml` or env): data paths, API base URL, rate limits, schedule.

---

## 4. Data Validation Layer

### 4.1 Purpose

Ensure only valid, consistent data reaches the processed layer; invalid runs do not overwrite or pollute downstream data.

### 4.2 Checks

- **Schema:** All required fields present; types match (e.g. Pydantic or equivalent).
- **Critical fields:** No nulls in `game_id`, `player_id`, `date`, and (for labeling) the field used to define "played" (e.g. `time_on_ice` or participation).
- **Duplicates:** Detect duplicate `(player_id, game_id)` and either deduplicate or fail and log.
- **Data types:** Numerics, dates, and IDs as expected.
- **DNP logic:** Ensure the field used for "played" is present and usable (e.g. TOI or box score presence).

### 4.3 Implementation

**File:** `src/validation.py` (or `data_validation.py`)

- Pydantic (or similar) schemas for raw and processed records.
- Assertions and explicit checks; log all anomalies (e.g. missing files, schema violations, duplicate counts).
- **On failure:** Log error, do not write to `data/processed/`. Optionally write invalid rows to `data/raw/failed/` or a log for inspection.

### 4.4 Unit Tests

**File:** `tests/test_validation.py`

- Valid payload passes.
- Missing required field fails.
- Wrong type fails.
- Duplicate `(player_id, game_id)` is caught.

---

## 5. Feature Engineering Pipeline

### 5.1 Principle

Every feature uses only information available **before** the game (no leakage). Opponent and team stats are point-in-time: e.g. "as of end of day before the game" (season-to-date or rolling), not future or same-day.

### 5.2 Implementation

**File:** `src/feature_engineering.py`

Reproducible pipeline (functions/classes), not ad hoc notebook code. **Input:** validated raw/processed game and player history. **Output:** one row per (player, game) with features and (optionally) label.

### 5.3 Feature List

**Recent performance (player, before game):**
- Goals in last 3 games (only games played).
- Assists in last 5 games.
- Rolling average points (e.g. 5-game window).
- Rolling shots on goal.
- Average time on ice (e.g. last 5 games).
- Trend: e.g. slope of points over last 5 games.
- Volatility: e.g. std of points over last N games.
- Simple fatigue proxy: e.g. games in last 7 days or back-to-back indicator.

**Opponent (as of day before game):**
- Opponent goals allowed per game (season-to-date or rolling).
- Opponent defensive ranking or similar.
- Opponent penalty kill % (if available).

**Game context:**
- Home vs away.
- Back-to-back (team or player).
- Days of rest.
- Season phase (e.g. early / mid / late by game number or date).

**Availability (optional but useful):**
- Played in previous game (0/1).
- Games played in last N (if available) to capture recent scratches.

All opponent and team stats must be computed as of the end of the day before the game (no same-day or future data).

### 5.4 Point-in-Time Guarantee

- Add a short comment or doc in code: "Opponent and team stats are computed as of end of day before game (season-to-date or rolling) to prevent leakage."
- **Unit tests:** e.g. assert no feature uses data from after the game date.

### 5.5 Cold Start (New / Low-History Players)

Choose and document one approach:

- **Option A – Minimum games:** Require at least N games played (e.g. 3–5) in the feature window; otherwise do not generate a prediction (API returns "insufficient history" or 404).
- **Option B – Fallback values:** Fill missing rolling stats with league-average or position-average values (from a small lookup table updated periodically).
- **Option C – Prior only:** For players with no history, return a simple prior (e.g. league-average P(≥1 point)) and document that the model is not used.

**Recommendation:** Option A for simplicity and clarity; document in API and README.

---

## 6. Dataset Construction

### 6.1 Labeled Dataset

For each player–game where the player played:
- **X:** Features computed using only data before that game (from the feature pipeline).
- **y:** 1 if player had ≥1 point in that game, 0 if they played and had 0 points.

Rows where the player did not play are excluded (not in the dataset).

### 6.2 Splits (Time-Based, No Shuffle)

- **Train:** Earlier seasons (e.g. 2018–19 through 2021–22).
- **Validation:** Next season (e.g. 2022–23) — used for hyperparameter tuning and model selection.
- **Test:** Following season (e.g. 2023–24) — used only for final evaluation and for comparing model versions in the registry; never for tuning.

Splits are stratified by time (by season or by date windows). Optionally stratify by target within each time window (e.g. similar positive rate in train/val/test) to keep class balance comparable.

### 6.3 Class Imbalance

- Report class balance (e.g. % positive) in train/val/test.
- Use stratified time-based splits so validation and test are not drastically different from train.
- In training: use `class_weight` (e.g. sklearn) or `scale_pos_weight` (XGBoost) to account for imbalance.
- Choose decision threshold using validation (e.g. maximize F1 or match target precision/recall); store this threshold with the model and use it in the API for binary decisions if needed.

---

## 7. Model Training

### 7.1 Implementation

**File:** `src/train.py`

- **Baseline:** Logistic regression.
- **Primary:** XGBoost (with class weighting as above).
- **Optional:** Random Forest for comparison.

Training uses validation set for early stopping / tuning; test set is never used during training.

### 7.2 Hyperparameter Tuning

Use GridSearch, RandomizedSearch, or Optuna over the validation set (or time-based CV within train). Do not use the test set for any tuning.

### 7.3 Artifacts to Save

**Directory:** `models/v1/` (then `v2/`, `v3/`, …)

- `model.pkl` (or equivalent) — serialized model (e.g. joblib).
- `features.json` — ordered list of feature names.
- `metrics.json` — validation (and optionally train) metrics; add test metrics after a single final evaluation run.
- `config.json` or `training_config.json` — hyperparameters, `scale_pos_weight`, and decision threshold (if fixed).
- `data_range.json` — e.g. train/val/test date ranges and (optionally) row counts.

### 7.4 Reproducibility

- Set random seeds.
- Record code version or commit (e.g. in `config.json` or registry).

---

## 8. Evaluation Metrics

### 8.1 Primary (Classification)

- ROC-AUC.
- Precision, Recall, F1 (at a chosen threshold).
- Confusion matrix.
- Calibration curve (predicted prob vs actual frequency).
- Precision/recall at a specific probability threshold (e.g. 0.5 or the tuned threshold).

### 8.2 Threshold

Choose one default threshold (e.g. via F1 or business constraint on validation). Store in model artifacts and use in API when a binary decision is required.

### 8.3 Explainability

- Feature importance (e.g. XGBoost native).
- SHAP (optional but recommended): summary plots or values logged or saved for key models.

### 8.4 Logging

- Validation metrics → `models/v1/metrics.json` (and similar for new versions).
- Test metrics → same file or a separate `test_metrics.json` after final evaluation; also used for registry comparison.

---

## 9. Model Versioning and Registry

### 9.1 Versioned Directories

Each training run can produce a new version: `models/v1/`, `models/v2/`, … Each contains: model artifact, `features.json`, `metrics.json`, config, data range, and (optionally) training date.

### 9.2 Registry

**File:** `model_registry.json` (at repo root or in `models/`)

- Current production model: e.g. v2.
- Previous versions: list with paths and metadata.
- Performance comparison: e.g. validation and test ROC-AUC (and optionally F1) per version.
- Training date and data range per version.

When promoting a new model, update the registry (e.g. set new `current_production`, append to history).

---

## 10. Feature Serving and Prediction API

### 10.1 Feature Serving Strategy

Decide and document one:

- **Option A – Pre-computed (recommended):** A scheduled job (e.g. daily or pre-game) runs the feature pipeline for "today's" or "next game" player–games and writes results to a feature table (e.g. CSV, Parquet, or SQLite). The API reads from this table.
- **Option B – On-demand:** The API loads raw/processed data and runs the feature pipeline for the requested `(player_id, game_id)`. Requires access to DB/files and may be slower; document data source and latency expectations.

**Recommendation:** Option A for production-like behavior and simpler API.

### 10.2 API

**File:** `api/main.py` (FastAPI)

**Endpoint:** `POST /predict`

**Input (example):**
```json
{
  "player_id": 8478402,
  "game_id": 2023020123
}
```

**Processing:**
- Resolve `(player_id, game_id)` to the appropriate feature row (from feature table or on-demand).
- If cold start (insufficient history): return 4xx or a structured response with `"insufficient_history": true` and no probability.
- Load production model from registry (version from config).
- Compute probability; optionally apply stored threshold for a binary decision.
- Return structured JSON.

**Output (example):**
```json
{
  "player_id": 8478402,
  "game_id": 2023020123,
  "prediction_probability": 0.72,
  "model_version": "v1",
  "binary_prediction": true
}
```

**Config:** Model version, feature table path, and cold-start policy read from config (e.g. `config/app.yaml` or env), not hardcoded.

### 10.3 Deployment

- **Minimum:** Local API + small demo script (e.g. `scripts/demo_predict.py`) that calls `/predict`.
- **Better:** Docker image; deploy to Render, Railway, AWS EC2, or GCP.
- **Optional:** Simple dashboard that displays predictions or model version.

### 10.4 Tests

**Contract test:** e.g. `tests/test_api.py` — valid request returns 200 and expected keys; invalid/missing player returns appropriate error; cold-start case returns defined response.

---

## 11. Monitoring Layer

### 11.1 Implementation

**File:** `monitoring/monitor.py` (or split into small modules)

### 11.2 What to Track

**Prediction distribution**
- Mean and variance of predicted probability (e.g. per day or per game night).
- Compare to training distribution (e.g. histogram or summary stats); flag drift if difference exceeds a threshold.

**Input feature drift**
- Compare current feature means (or quantiles) to training feature means.
- Flag if shift exceeds tolerance (e.g. by Z-score or simple % change).

**Model performance (when ground truth available)**
- **Daily (or post-game-day) evaluation job:**
  - Load predictions that were made for past games (e.g. stored when `/predict` was called, or recomputed from stored features).
  - Join with actual outcomes (from box scores).
  - Compute rolling precision, recall, F1, and optionally calibration.
  - Log to CSV, SQLite, or a small dashboard.

### 11.3 Logging and Alerts

- Log to CSV/SQLite and (optionally) a simple dashboard.
- Config: paths, drift thresholds, and which metrics to compute.

---

## 12. Retraining Logic

### 12.1 Implementation

**File:** `src/retrain.py`

### 12.2 Triggers

Retrain when any of:
- **Performance:** Rolling precision/recall or ROC-AUC (from the daily evaluation job) drops below a configured threshold.
- **Drift:** Feature or prediction distribution drift exceeds tolerance.
- **Schedule:** e.g. monthly retraining (configurable).

### 12.3 Retraining Steps

1. Pull latest data; run ingestion and validation.
2. Rebuild features and dataset (same split logic: train/val/test).
3. Retrain model (same training code); evaluate on validation and test.
4. **Promotion criterion:** New model is promoted only if:
   - Validation metric (e.g. ROC-AUC or F1) is better than current production, **and**
   - Test metric does not degrade beyond a small tolerance (e.g. no drop in test ROC-AUC or F1 beyond X%).

If promoted: save as new version (e.g. `models/v3/`), update `model_registry.json` (set new production, record metrics).

If not promoted: keep current production; optionally log "retrain run not promoted" with reason.

### 12.4 Config

Store thresholds (performance floor, drift tolerance, test degradation tolerance) and schedule in config (e.g. `config/retrain.yaml`).

---

## 13. Configuration and Repo Structure

### 13.1 Configuration

- **Central config:** e.g. `config/config.yaml` (or split: `config/ingestion.yaml`, `config/train.yaml`, `config/api.yaml`, `config/retrain.yaml`).
- **Contents:** paths (raw, processed, models, feature table), model version, API keys (or env var names), retrain thresholds, and drift tolerances.
- **Secrets:** Use env vars (e.g. `API_KEY`); never commit secrets.

### 13.2 Project Structure

```
ml_nhl_prediction/
├── config/
│   ├── config.yaml          # or ingestion.yaml, train.yaml, api.yaml, retrain.yaml
│   └── .gitignore           # ignore secrets if any file holds them
├── data/
│   ├── raw/
│   │   └── YYYY_MM_DD/
│   ├── processed/
│   └── features/            # optional: feature store output
├── src/
│   ├── data_ingestion.py
│   ├── validation.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── retrain.py
├── api/
│   └── main.py
├── monitoring/
│   └── monitor.py
├── models/
│   ├── v1/
│   ├── v2/
│   └── ...
├── scripts/
│   ├── run_ingestion.sh     # or .py
│   ├── run_retrain.sh
│   └── demo_predict.py
├── tests/
│   ├── test_validation.py
│   ├── test_feature_engineering.py  # e.g. no future data
│   └── test_api.py
├── notebooks/
│   ├── eda.ipynb
│   └── shap_analysis.ipynb  # optional
├── model_registry.json
├── requirements.txt
└── README.md
```

---

## 14. Testing Summary

- **Validation:** Schema, types, duplicates (see Section 4.4).
- **Feature engineering:** No use of future data; point-in-time opponent stats (see Section 5.4).
- **API:** Contract and cold-start behavior (see Section 10.4).
- **Optional:** integration test that runs a minimal train → save → load → predict flow.

---

## 15. Documentation (README)

- **Problem:** Target (≥1 point given they play), DNP exclusion, cold-start policy.
- **Data:** Source (NHL API or other), how "played" is defined, raw and processed layout.
- **Splits:** Train/val/test seasons and usage (tuning vs final evaluation).
- **How to run:** Ingestion, validation, feature job, training, API, monitoring, retrain.
- **Config:** Where config files live and main options.
- **Model registry:** Where it is and how promotion works.

---

This plan is the single source of truth for the project with DNP handling, class imbalance, train/val/test, feature serving, cold start, promotion criterion, monitoring with a daily evaluation job, and config/tests/docs all included. You can tick off sections as you implement them.
