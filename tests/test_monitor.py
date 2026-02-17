"""Tests for monitoring module (Section 11)."""

from pathlib import Path

import pandas as pd
import pytest

from monitoring.monitor import (
    META_COLS,
    _get_monitoring_dir,
    _load_config,
    run_feature_drift,
    run_prediction_distribution,
    run_daily_evaluation,
)


def test_meta_cols_contains_expected():
    """META_COLS includes required metadata columns."""
    assert "game_id" in META_COLS
    assert "player_id" in META_COLS
    assert "label" in META_COLS
    assert "game_date" in META_COLS


def test_load_config_returns_dict():
    """_load_config returns config dict with paths."""
    config = _load_config()
    assert isinstance(config, dict)
    assert "paths" in config


def test_get_monitoring_dir_creates_dir():
    """_get_monitoring_dir returns path and creates directory."""
    config = _load_config()
    monitor_dir = _get_monitoring_dir(config)
    assert monitor_dir.exists()
    assert monitor_dir.is_dir()


def test_run_feature_drift_requires_features():
    """run_feature_drift raises or skips when no feature data."""
    root = Path(__file__).resolve().parent.parent
    feat_dir = root / "data" / "features"
    parquets = list(feat_dir.glob("features_*.parquet"))
    parquets = [p for p in parquets if "_meta" not in p.name]
    if not parquets:
        pytest.skip("No feature parquet found")
    result = run_feature_drift(features_path=parquets[0], recent_days=90)
    assert isinstance(result, dict)
    assert "drift_detected" in result


def test_run_prediction_distribution_requires_model():
    """run_prediction_distribution needs model and features."""
    root = Path(__file__).resolve().parent.parent
    if not (root / "model_registry.json").exists():
        pytest.skip("No model registry")
    feat_dir = root / "data" / "features"
    parquets = list(feat_dir.glob("features_*.parquet"))
    parquets = [p for p in parquets if "_meta" not in p.name]
    if not parquets:
        pytest.skip("No feature parquet found")
    result = run_prediction_distribution(features_path=parquets[0], recent_days=90)
    assert isinstance(result, dict)
    assert "drift_detected" in result
    assert "train_pred_mean" in result or "recent_pred_mean" in result


def test_run_daily_evaluation_requires_model():
    """run_daily_evaluation needs model and features with labels."""
    root = Path(__file__).resolve().parent.parent
    if not (root / "model_registry.json").exists():
        pytest.skip("No model registry")
    feat_dir = root / "data" / "features"
    parquets = list(feat_dir.glob("features_*.parquet"))
    parquets = [p for p in parquets if "_meta" not in p.name]
    if not parquets:
        pytest.skip("No feature parquet found")
    result = run_daily_evaluation(features_path=parquets[0], eval_days=90)
    assert isinstance(result, dict)
    if result:
        assert "roc_auc" in result or "f1" in result
