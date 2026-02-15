"""
Model evaluation (Section 8).

ROC-AUC, precision, recall, F1, confusion matrix.
Logs metrics to model artifact directory.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model_path: str | Path,
    test_data_path: str | Path,
    output_path: str | Path | None = None,
    threshold: float | None = None,
) -> dict:
    """
    Evaluate model on test set.
    Returns metrics dict. Optionally appends test metrics to output_path.
    """
    model_path = Path(model_path)
    test_data_path = Path(test_data_path)

    model = joblib.load(model_path)
    df = pd.read_parquet(test_data_path)

    # Load feature names from model dir
    features_file = model_path.parent / "features.json"
    if features_file.exists():
        with open(features_file) as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names", [])
    else:
        meta_cols = ["game_id", "player_id", "team_id", "opponent_team_id", "game_date", "season", "label"]
        feature_names = [c for c in df.columns if c not in meta_cols]

    X = df[feature_names].fillna(0)
    y_true = df["label"].values

    if threshold is None:
        config_file = model_path.parent / "training_config.json"
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f)
            threshold = cfg.get("threshold", 0.5)
        else:
            threshold = 0.5

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"test": metrics}, f, indent=2)

    return metrics
