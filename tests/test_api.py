"""API contract tests (Section 10.4).

TestClient must be used with a context manager so lifespan (startup) runs.
"""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


def _get_sample_player_game():
    """Get a (player_id, game_id) that exists in the feature table."""
    root = Path(__file__).resolve().parent.parent
    feat_dir = root / "data" / "features"
    parquets = sorted(feat_dir.glob("features_*.parquet"), reverse=True)
    parquets = [p for p in parquets if "_meta" not in p.name]
    if not parquets:
        pytest.skip("No feature parquet found")
    df = pd.read_parquet(parquets[0])
    row = df[["player_id", "game_id"]].drop_duplicates().iloc[0]
    return int(row["player_id"]), int(row["game_id"])


def test_health_returns_200():
    """Health endpoint returns 200."""
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_valid_request_returns_200_and_expected_keys():
    """Valid (player_id, game_id) returns 200 and expected response keys."""
    player_id, game_id = _get_sample_player_game()
    with TestClient(app) as client:
        resp = client.post("/predict", json={"player_id": player_id, "game_id": game_id})
    assert resp.status_code == 200
    data = resp.json()
    assert "player_id" in data
    assert "game_id" in data
    assert "prediction_probability" in data
    assert "model_version" in data
    assert "binary_prediction" in data
    assert data["player_id"] == player_id
    assert data["game_id"] == game_id
    assert 0 <= data["prediction_probability"] <= 1
    assert isinstance(data["binary_prediction"], bool)


def test_invalid_player_returns_404():
    """Non-existent (player_id, game_id) returns 404 with insufficient_history."""
    with TestClient(app) as client:
        resp = client.post("/predict", json={"player_id": 999999999, "game_id": 999999999})
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data
    detail = data["detail"]
    if isinstance(detail, dict):
        assert detail.get("insufficient_history") is True
    else:
        assert "insufficient_history" in str(detail).lower()


def test_cold_start_returns_defined_response():
    """Cold start / not found returns structured 404."""
    with TestClient(app) as client:
        resp = client.post("/predict", json={"player_id": 0, "game_id": 0})
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data
