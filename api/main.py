"""
Prediction API (Section 10).

FastAPI endpoint: POST /predict
Returns P(≥1 point | player plays) for given player_id, game_id.
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import requests
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


class PredictRequest(BaseModel):
    player_id: int
    game_id: int


class PredictResponse(BaseModel):
    player_id: int
    game_id: int
    prediction_probability: float
    model_version: str
    binary_prediction: bool
    insufficient_history: bool = False


def _load_app_state():
    """Load model, features, and config on startup."""
    root = Path(__file__).resolve().parent.parent
    with open(root / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    api_cfg = config.get("api", {})
    # Prefer registry current_production; fallback to config
    registry_path = root / "model_registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
        model_version = registry.get("current_production") or api_cfg.get("model_version", "v1")
    else:
        model_version = api_cfg.get("model_version", "v1")
    models_dir = root / config["paths"]["models"]
    model_dir = models_dir / model_version

    model = joblib.load(model_dir / "model.pkl")
    with open(model_dir / "features.json") as f:
        feature_names = json.load(f)["feature_names"]
    with open(model_dir / "training_config.json") as f:
        training_cfg = json.load(f)
    threshold = training_cfg.get("threshold", 0.5)

    feat_dir = root / config["paths"]["features"]
    parquets = sorted(feat_dir.glob("features_*.parquet"), reverse=True)
    parquets = [p for p in parquets if "_meta" not in p.name]
    if not parquets:
        raise FileNotFoundError(f"No feature parquet in {feat_dir}")
    features_df = pd.read_parquet(parquets[0])
    # Index for fast lookup
    features_df = features_df.set_index(["player_id", "game_id"])

    return {
        "model": model,
        "feature_names": feature_names,
        "threshold": threshold,
        "model_version": model_version,
        "features_df": features_df,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_state = _load_app_state()
    yield
    app.state.model_state = None


app = FastAPI(
    title="NHL Point Prediction API",
    description="Predict probability a player records ≥1 point in a game (given they play).",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    """Health check."""
    state = request.app.state.model_state
    return {"status": "ok", "model_version": state["model_version"] if state else None}


# In-memory cache for NHL API lookups (avoids repeated calls)
_player_cache: dict[int, str] = {}
_game_cache: dict[int, str] = {}
_CACHE_MAX = 500

NHL_BASE = "https://api-web.nhle.com/v1"


def _fetch_player_name(player_id: int) -> str:
    if player_id in _player_cache:
        return _player_cache[player_id]
    try:
        r = requests.get(f"{NHL_BASE}/player/{player_id}/landing", timeout=5)
        if r.status_code == 200:
            d = r.json()
            first = d.get("firstName", {})
            last = d.get("lastName", {})
            name = f"{first.get('default', '')} {last.get('default', '')}".strip()
            if name:
                if len(_player_cache) >= _CACHE_MAX:
                    _player_cache.pop(next(iter(_player_cache)))
                _player_cache[player_id] = name
                return name
    except Exception:
        pass
    return str(player_id)


def _fetch_game_info(game_id: int) -> str:
    if game_id in _game_cache:
        return _game_cache[game_id]
    try:
        r = requests.get(f"{NHL_BASE}/gamecenter/{game_id}/landing", timeout=5)
        if r.status_code == 200:
            d = r.json()
            away = d.get("awayTeam", {}).get("abbrev", "?")
            home = d.get("homeTeam", {}).get("abbrev", "?")
            date = d.get("gameDate", "")
            info = f"{away} @ {home} {date}"
            if len(_game_cache) >= _CACHE_MAX:
                _game_cache.pop(next(iter(_game_cache)))
            _game_cache[game_id] = info
            return info
    except Exception:
        pass
    return str(game_id)


@app.get("/sample")
def sample(request: Request, limit: int = 30, expand: bool = False):
    """Return sample (player_id, game_id) pairs from the feature table. Use expand=1 for player/game names."""
    state = request.app.state.model_state
    if not state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = state["features_df"].reset_index()
    unique = df[["player_id", "game_id"]].drop_duplicates()
    n = min(limit, len(unique))
    sample_df = unique.sample(n=n, random_state=42)
    rows = sample_df.to_dict(orient="records")

    if expand:
        for row in rows:
            row["player_name"] = _fetch_player_name(int(row["player_id"]))
            row["game_info"] = _fetch_game_info(int(row["game_id"]))
            time.sleep(0.15)  # rate limit

    return rows


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, payload: PredictRequest):
    """
    Predict P(≥1 point | player plays).
    Returns 404 with insufficient_history if player-game not in feature table.
    """
    state = request.app.state.model_state
    if not state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        row = state["features_df"].loc[(payload.player_id, payload.game_id)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
    except (KeyError, TypeError):
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Player-game not found in feature table",
                "insufficient_history": True,
            },
        )

    X = state["feature_names"]
    feature_values = [row[c] for c in X]
    X_arr = np.array([feature_values], dtype=float)
    X_arr = np.nan_to_num(X_arr, nan=0.0)

    prob = float(state["model"].predict_proba(X_arr)[0, 1])
    binary = prob >= state["threshold"]

    return PredictResponse(
        player_id=payload.player_id,
        game_id=payload.game_id,
        prediction_probability=round(prob, 4),
        model_version=state["model_version"],
        binary_prediction=binary,
        insufficient_history=False,
    )
