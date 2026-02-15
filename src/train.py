"""
Model training (Section 7).

Baseline: Logistic regression. Primary: XGBoost.
Uses validation set for tuning; test set reserved for final evaluation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def _get_feature_names_and_data(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Extract feature columns and X DataFrame."""
    meta_cols = ["game_id", "player_id", "team_id", "opponent_team_id", "game_date", "season", "label"]
    feature_names = [c for c in df.columns if c not in meta_cols]
    X = df[feature_names].copy()
    return feature_names, X


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute XGBoost scale_pos_weight = n_neg / n_pos."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def _find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 on validation."""
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.7, 0.02):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute ROC-AUC, precision, recall, F1, confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
    }


def train(
    features_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_version: str | None = None,
    update_registry: bool = True,
) -> str:
    """
    Train model and save to models/vN/.
    Returns model version (e.g. 'v1').
    """
    config = _load_config()
    root = Path(__file__).resolve().parent.parent
    cfg = config.get("training", {})
    train_seasons = cfg.get("train_seasons", ["20182019", "20192020", "20202021", "20212022"])
    val_season = cfg.get("val_season", "20222023")
    test_season = cfg.get("test_season", "20232024")
    random_seed = cfg.get("random_seed", 42)

    if features_path is None:
        feat_dir = root / config["paths"]["features"]
        parquets = sorted(feat_dir.glob("features_*.parquet"), reverse=True)
        parquets = [p for p in parquets if "_meta" not in p.name]
        if not parquets:
            raise FileNotFoundError(f"No feature parquet files in {feat_dir}")
        features_path = parquets[0]

    features_path = Path(features_path)
    logger.info("Loading features from %s", features_path)

    df = pd.read_parquet(features_path)
    feature_names, X = _get_feature_names_and_data(df)
    y = df["label"].values

    # Fill NaN with 0 (e.g. from rolling stats with insufficient history)
    X = X.fillna(0)

    # Time-based splits by season
    train_mask = df["season"].astype(str).isin([str(s) for s in train_seasons])
    val_mask = df["season"].astype(str) == str(val_season)
    test_mask = df["season"].astype(str) == str(test_season)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))
    logger.info("Train label %% pos: %.2f, Val: %.2f, Test: %.2f",
                y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100)

    scale_pos_weight = _compute_scale_pos_weight(pd.Series(y_train))
    logger.info("scale_pos_weight: %.3f", scale_pos_weight)

    # Resolve version and output dir
    models_dir = root / config["paths"]["models"]
    if model_version is None:
        existing = [d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
        versions = [int(v[1:]) for v in existing if v[1:].isdigit()]
        next_v = max(versions, default=0) + 1
        model_version = f"v{next_v}"
    if output_dir is None:
        output_dir = models_dir / model_version
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(random_seed)

    # --- Baseline: Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=random_seed, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_val_prob = lr.predict_proba(X_val)[:, 1]
    lr_thresh = _find_optimal_threshold(y_val, lr_val_prob)
    lr_val_metrics = _compute_metrics(y_val, lr_val_prob, lr_thresh)
    logger.info("Logistic Regression val ROC-AUC: %.4f, F1: %.4f", lr_val_metrics["roc_auc"], lr_val_metrics["f1"])

    # --- Primary: XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        eval_metric="logloss",
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_val_prob = xgb.predict_proba(X_val)[:, 1]
    xgb_thresh = _find_optimal_threshold(y_val, xgb_val_prob)
    xgb_val_metrics = _compute_metrics(y_val, xgb_val_prob, xgb_thresh)
    logger.info("XGBoost val ROC-AUC: %.4f, F1: %.4f", xgb_val_metrics["roc_auc"], xgb_val_metrics["f1"])

    # Use XGBoost as production model (better performer)
    model = xgb
    threshold = xgb_thresh
    val_metrics = xgb_val_metrics

    # Test evaluation (single run, for registry)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_prob, threshold)

    # --- Save artifacts ---
    joblib.dump(model, output_dir / "model.pkl")

    with open(output_dir / "features.json", "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)

    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "train_class_balance_pct": float(y_train.mean() * 100),
        "val_class_balance_pct": float(y_val.mean() * 100),
        "test_class_balance_pct": float(y_test.mean() * 100),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    training_config = {
        "model_type": "xgboost",
        "scale_pos_weight": scale_pos_weight,
        "threshold": threshold,
        "random_seed": random_seed,
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    data_range = {
        "train_seasons": train_seasons,
        "val_season": val_season,
        "test_season": test_season,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }
    with open(output_dir / "data_range.json", "w") as f:
        json.dump(data_range, f, indent=2)

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    # Update registry (unless caller will handle promotion, e.g. retrain.py)
    if update_registry:
        registry_path = root / "model_registry.json"
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"current_production": None, "versions": []}
        version_entry = {
            "version": model_version,
            "path": str(output_dir.relative_to(root)),
            "val_roc_auc": val_metrics["roc_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "test_f1": test_metrics["f1"],
            "trained_at": datetime.now().isoformat(),
        }
        if model_version not in [v.get("version") for v in registry.get("versions", [])]:
            registry.setdefault("versions", []).append(version_entry)
        registry["current_production"] = model_version
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    logger.info("Saved to %s. Val ROC-AUC: %.4f, Test ROC-AUC: %.4f",
                output_dir, val_metrics["roc_auc"], test_metrics["roc_auc"])

    return model_version


if __name__ == "__main__":
    train()
