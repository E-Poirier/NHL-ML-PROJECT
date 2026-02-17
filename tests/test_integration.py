"""Integration tests: minimal train → save → load → predict flow."""

import tempfile
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.train import train


def test_train_save_load_predict():
    """Train model, save to temp dir, load, and run predict_proba on a sample."""
    root = Path(__file__).resolve().parent.parent
    feat_dir = root / "data" / "features"
    parquets = sorted(feat_dir.glob("features_*.parquet"), reverse=True)
    parquets = [p for p in parquets if "_meta" not in p.name]
    if not parquets:
        pytest.skip("No feature parquet found")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "models" / "v_test"
        out_dir.mkdir(parents=True)

        # Train to temp dir, do not update registry
        version = train(
            features_path=parquets[0],
            output_dir=out_dir,
            model_version="v_test",
            update_registry=False,
        )
        assert version == "v_test"

        # Load model
        model = joblib.load(out_dir / "model.pkl")

        # Load features for prediction
        df = pd.read_parquet(parquets[0])
        meta_cols = ["game_id", "player_id", "team_id", "opponent_team_id", "game_date", "season", "label"]
        feature_names = [c for c in df.columns if c not in meta_cols]
        X = df[feature_names].fillna(0).head(5)

        # Predict
        probs = model.predict_proba(X)[:, 1]
        assert len(probs) == 5
        assert all(0 <= p <= 1 for p in probs)
