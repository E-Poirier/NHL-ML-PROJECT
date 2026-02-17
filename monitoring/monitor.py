"""
Monitoring layer (Section 11).

Tracks: prediction distribution, input feature drift, model performance.
Logs to CSV for retrain triggers and optional dashboard.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

META_COLS = [
    "game_id",
    "player_id",
    "team_id",
    "opponent_team_id",
    "game_date",
    "season",
    "label",
]


def _load_config() -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def _get_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _get_monitoring_dir(config: dict) -> Path:
    root = _get_root()
    monitor_path = root / config["paths"].get("monitoring", "monitoring")
    monitor_path.mkdir(parents=True, exist_ok=True)
    return monitor_path


def _load_model_and_features(config: dict) -> tuple:
    """Load production model, feature names, and threshold from registry."""
    root = _get_root()
    with open(root / "model_registry.json") as f:
        registry = json.load(f)
    prod = registry.get("current_production")
    if not prod:
        raise ValueError("No production model in registry")
    model_dir = root / config["paths"]["models"] / prod
    model = joblib.load(model_dir / "model.pkl")
    with open(model_dir / "features.json") as f:
        feature_names = json.load(f)["feature_names"]
    with open(model_dir / "training_config.json") as f:
        threshold = json.load(f).get("threshold", 0.5)
    return model, feature_names, threshold, prod


def _load_features_data(config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load latest feature parquet and feature names."""
    root = _get_root()
    feat_dir = root / config["paths"]["features"]
    parquets = sorted(feat_dir.glob("features_*.parquet"), reverse=True)
    parquets = [p for p in parquets if "_meta" not in p.name]
    if not parquets:
        raise FileNotFoundError(f"No feature parquet in {feat_dir}")
    df = pd.read_parquet(parquets[0])
    feature_names = [c for c in df.columns if c not in META_COLS]
    return df, feature_names


def run_prediction_distribution(
    features_path: str | Path | None = None,
    recent_days: int = 30,
) -> dict:
    """
    Compute prediction distribution for recent data vs training.
    Log mean/variance; flag drift if difference exceeds threshold.
    """
    config = _load_config()
    monitor_dir = _get_monitoring_dir(config)
    drift_z = config.get("retrain", {}).get("drift_tolerance_zscore", 3.0)

    if features_path:
        df = pd.read_parquet(features_path)
    else:
        df, _ = _load_features_data(config)

    model, feature_names, _, _ = _load_model_and_features(config)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Training baseline: use train seasons from config
    train_seasons = config.get("training", {}).get("train_seasons", [])
    train_seasons = [str(s) for s in train_seasons]
    train_df = df[df["season"].astype(str).isin(train_seasons)]
    X_train = train_df[feature_names].fillna(0)
    train_probs = model.predict_proba(X_train)[:, 1]

    # Recent: last N days
    cutoff = df["game_date"].max() - pd.Timedelta(days=recent_days)
    recent_df = df[df["game_date"] >= cutoff]
    if recent_df.empty:
        logger.warning("No recent data for prediction distribution")
        return {"pred_mean": None, "pred_std": None, "drift_detected": False}

    X_recent = recent_df[feature_names].fillna(0)
    recent_probs = model.predict_proba(X_recent)[:, 1]

    train_mean, train_std = float(np.mean(train_probs)), float(np.std(train_probs)) or 1e-9
    recent_mean, recent_std = float(np.mean(recent_probs)), float(np.std(recent_probs))

    # Simple drift: if recent mean differs from train by > Z std
    z_diff = abs(recent_mean - train_mean) / (train_std or 1e-9)
    drift_detected = z_diff > drift_z

    result = {
        "date": datetime.now().isoformat()[:10],
        "train_pred_mean": train_mean,
        "train_pred_std": train_std,
        "recent_pred_mean": recent_mean,
        "recent_pred_std": recent_std,
        "recent_n": len(recent_df),
        "z_score": z_diff,
        "drift_detected": drift_detected,
    }

    out_file = monitor_dir / "prediction_distribution.csv"
    row = pd.DataFrame([{k: v for k, v in result.items()}])
    if out_file.exists():
        hist = pd.read_csv(out_file)
        hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row
    hist.to_csv(out_file, index=False)

    logger.info(
        "Prediction dist: recent mean=%.4f (train=%.4f), z=%.2f, drift=%s",
        recent_mean,
        train_mean,
        z_diff,
        drift_detected,
    )
    return result


def run_feature_drift(
    features_path: str | Path | None = None,
    recent_days: int = 30,
) -> dict:
    """
    Compare current feature means to training feature means.
    Flag if any feature shifts beyond Z-score tolerance.
    """
    config = _load_config()
    monitor_dir = _get_monitoring_dir(config)
    drift_z = config.get("retrain", {}).get("drift_tolerance_zscore", 3.0)

    if features_path:
        df = pd.read_parquet(features_path)
        feature_names = [c for c in df.columns if c not in META_COLS]
    else:
        df, feature_names = _load_features_data(config)
        feature_names = [c for c in feature_names if c in df.columns]

    df["game_date"] = pd.to_datetime(df["game_date"])
    train_seasons = [str(s) for s in config.get("training", {}).get("train_seasons", [])]
    train_df = df[df["season"].astype(str).isin(train_seasons)]
    cutoff = df["game_date"].max() - pd.Timedelta(days=recent_days)
    recent_df = df[df["game_date"] >= cutoff]

    if recent_df.empty:
        logger.warning("No recent data for feature drift")
        return {"drift_detected": False, "drifted_features": []}

    X_train = train_df[feature_names].fillna(0)
    X_recent = recent_df[feature_names].fillna(0)
    train_means = X_train.mean()
    train_stds = X_train.std().replace(0, 1e-9)
    recent_means = X_recent.mean()
    z_scores = ((recent_means - train_means) / train_stds).abs()
    drifted = z_scores[z_scores > drift_z].to_dict()
    drift_detected = len(drifted) > 0

    result = {
        "date": datetime.now().isoformat()[:10],
        "drift_detected": drift_detected,
        "drifted_features": list(drifted.keys()),
        "max_z_score": float(z_scores.max()) if len(z_scores) > 0 else 0,
    }

    out_file = monitor_dir / "drift_flags.csv"
    row = pd.DataFrame([{k: json.dumps(v) if isinstance(v, list) else v for k, v in result.items()}])
    if out_file.exists():
        hist = pd.read_csv(out_file)
        hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row
    hist.to_csv(out_file, index=False)

    logger.info("Feature drift: detected=%s, drifted=%s", drift_detected, result["drifted_features"])
    return result


def run_daily_evaluation(
    features_path: str | Path | None = None,
    eval_days: int = 7,
) -> dict:
    """
    Evaluate model on recent games (predictions vs actual outcomes).
    Compute precision, recall, F1, ROC-AUC; append to eval_metrics.csv.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    config = _load_config()
    monitor_dir = _get_monitoring_dir(config)

    if features_path:
        df = pd.read_parquet(features_path)
    else:
        df, feature_names = _load_features_data(config)

    model, feature_names, threshold, _ = _load_model_and_features(config)
    feature_names = [c for c in feature_names if c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"])

    cutoff = df["game_date"].max() - pd.Timedelta(days=eval_days)
    eval_df = df[df["game_date"] >= cutoff]
    if eval_df.empty or "label" not in eval_df.columns:
        logger.warning("No recent labeled data for daily evaluation")
        return {}

    X = eval_df[feature_names].fillna(0)
    y_true = eval_df["label"].values
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    result = {
        "date": datetime.now().isoformat()[:10],
        "eval_days": eval_days,
        "n_samples": len(eval_df),
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    out_file = monitor_dir / "eval_metrics.csv"
    row = pd.DataFrame([result])
    if out_file.exists():
        hist = pd.read_csv(out_file)
        hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row
    hist.to_csv(out_file, index=False)

    logger.info(
        "Daily eval (%d days, n=%d): ROC-AUC=%.4f, F1=%.4f",
        eval_days,
        len(eval_df),
        roc_auc,
        f1,
    )
    return result


def run_all(
    features_path: str | Path | None = None,
    recent_days: int = 30,
    eval_days: int = 7,
) -> dict:
    """Run all monitoring checks: prediction dist, feature drift, daily evaluation."""
    results = {}
    try:
        results["prediction_distribution"] = run_prediction_distribution(
            features_path=features_path, recent_days=recent_days
        )
    except Exception as e:
        logger.error("Prediction distribution failed: %s", e)
        results["prediction_distribution"] = {"error": str(e)}

    try:
        results["feature_drift"] = run_feature_drift(
            features_path=features_path, recent_days=recent_days
        )
    except Exception as e:
        logger.error("Feature drift failed: %s", e)
        results["feature_drift"] = {"error": str(e)}

    try:
        results["daily_eval"] = run_daily_evaluation(
            features_path=features_path, eval_days=eval_days
        )
    except Exception as e:
        logger.error("Daily evaluation failed: %s", e)
        results["daily_eval"] = {"error": str(e)}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run monitoring (Section 11)")
    parser.add_argument("--features", type=str, help="Path to features parquet (optional)")
    parser.add_argument("--recent-days", type=int, default=30, help="Days for drift/recent window")
    parser.add_argument("--eval-days", type=int, default=7, help="Days for daily evaluation window")
    parser.add_argument("--pred-only", action="store_true", help="Only prediction distribution")
    parser.add_argument("--drift-only", action="store_true", help="Only feature drift")
    parser.add_argument("--eval-only", action="store_true", help="Only daily evaluation")
    args = parser.parse_args()

    feat_path = Path(args.features) if args.features else None

    if args.pred_only:
        run_prediction_distribution(features_path=feat_path, recent_days=args.recent_days)
    elif args.drift_only:
        run_feature_drift(features_path=feat_path, recent_days=args.recent_days)
    elif args.eval_only:
        run_daily_evaluation(features_path=feat_path, eval_days=args.eval_days)
    else:
        run_all(
            features_path=feat_path,
            recent_days=args.recent_days,
            eval_days=args.eval_days,
        )
