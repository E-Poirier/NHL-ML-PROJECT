"""
Retraining logic (Section 12).

Triggers: performance drop, drift, or schedule.
Promotion only if validation improves and test does not degrade.
"""

import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def _get_registry() -> dict:
    root = Path(__file__).resolve().parent.parent
    registry_path = root / "model_registry.json"
    if not registry_path.exists():
        return {"current_production": None, "versions": []}
    return json.loads(registry_path.read_text())


def _get_production_metrics(registry: dict) -> dict | None:
    """Get metrics for current production model."""
    prod = registry.get("current_production")
    if not prod:
        return None
    for v in registry.get("versions", []):
        if v.get("version") == prod:
            return v
    return None


def check_retrain_needed(
    force_schedule: bool = False,
) -> tuple[bool, str]:
    """
    Check if retraining is triggered.

    Triggers: performance floor, drift (from monitoring), or schedule.
    Returns (should_retrain, reason).
    """
    config = _load_config()
    root = Path(__file__).resolve().parent.parent
    retrain_cfg = config.get("retrain", {})
    schedule = retrain_cfg.get("schedule", "monthly")

    # Schedule trigger: last trained date
    registry = _get_registry()
    prod_metrics = _get_production_metrics(registry)
    if prod_metrics and schedule in ("monthly", "weekly"):
        try:
            from datetime import datetime, timedelta

            trained_at = prod_metrics.get("trained_at")
            if trained_at:
                # Parse and strip timezone for comparison
                last = datetime.fromisoformat(trained_at.replace("Z", "").split("+")[0])
                if last.tzinfo:
                    last = last.replace(tzinfo=None)
                now = datetime.now()
                delta = now - last
                if schedule == "monthly" and delta > timedelta(days=30):
                    return True, f"Schedule: last trained {delta.days} days ago (monthly)"
                if schedule == "weekly" and delta > timedelta(days=7):
                    return True, f"Schedule: last trained {delta.days} days ago (weekly)"
        except Exception as e:
            logger.warning("Could not parse trained_at for schedule check: %s", e)

    # Performance trigger: read from monitoring if available
    eval_path = root / config["paths"].get("monitoring", "monitoring") / "eval_metrics.csv"
    if eval_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(eval_path)
            if len(df) > 0:
                floor = retrain_cfg.get("performance_floor_roc_auc", 0.55)
                recent = df.tail(7)  # Last 7 days
                if "roc_auc" in df.columns and recent["roc_auc"].mean() < floor:
                    return True, f"Performance: rolling ROC-AUC {recent['roc_auc'].mean():.4f} < {floor}"
        except Exception as e:
            logger.debug("Could not check performance from monitoring: %s", e)

    # Drift trigger: read from monitoring if available
    drift_path = root / config["paths"].get("monitoring", "monitoring") / "drift_flags.csv"
    if drift_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(drift_path)
            if len(df) > 0 and "drift_detected" in df.columns and df["drift_detected"].iloc[-1]:
                return True, "Drift: feature or prediction drift exceeds tolerance"
        except Exception as e:
            logger.debug("Could not check drift from monitoring: %s", e)

    if force_schedule:
        return True, "Forced by schedule/on_demand"

    return False, "No retrain trigger"


def run_retrain(
    skip_ingestion: bool = False,
    force_promotion: bool = False,
) -> bool:
    """
    Run full retrain pipeline.

    1. Pull latest data (unless skip_ingestion)
    2. Validate and process
    3. Rebuild features
    4. Retrain model
    5. Promote only if validation improves and test does not degrade

    Returns True if new model promoted to production.
    """
    config = _load_config()
    root = Path(__file__).resolve().parent.parent
    retrain_cfg = config.get("retrain", {})
    perf_floor = retrain_cfg.get("performance_floor_roc_auc", 0.55)
    degrade_tol = retrain_cfg.get("test_degradation_tolerance_pct", 0.05)

    # 1. Ingestion (optional)
    if not skip_ingestion:
        logger.info("Step 1: Running ingestion...")
        from src.data_ingestion import run_ingestion

        run_ingestion()
    else:
        logger.info("Step 1: Skipping ingestion (skip_ingestion=True)")

    # 2. Validation
    logger.info("Step 2: Validating and processing...")
    from src.validation import validate_and_process

    if not validate_and_process():
        logger.error("Validation failed; aborting retrain")
        return False

    # 3. Feature engineering
    logger.info("Step 3: Building features...")
    from src.feature_engineering import build_features

    build_features()

    # 4. Train (do not update registry yet)
    logger.info("Step 4: Training model...")
    from src.train import train

    model_version = train(update_registry=False)

    # 5. Promotion criterion
    model_dir = root / config["paths"]["models"] / model_version
    with open(model_dir / "metrics.json") as f:
        new_metrics = json.load(f)
    new_val_auc = new_metrics["validation"]["roc_auc"]
    new_test_auc = new_metrics["test"]["roc_auc"]
    new_val_f1 = new_metrics["validation"]["f1"]
    new_test_f1 = new_metrics["test"]["f1"]

    registry = _get_registry()
    prod_metrics = _get_production_metrics(registry)

    if prod_metrics is None:
        # No current production; promote
        logger.info("No current production; promoting %s", model_version)
        _promote_model(root, config, model_version, new_metrics)
        return True

    prod_val_auc = prod_metrics.get("val_roc_auc", 0)
    prod_test_auc = prod_metrics.get("test_roc_auc", 0)
    prod_val_f1 = prod_metrics.get("val_f1", 0)
    prod_test_f1 = prod_metrics.get("test_f1", 0)

    if force_promotion:
        logger.info("Force promotion enabled; promoting %s", model_version)
        _promote_model(root, config, model_version, new_metrics)
        return True

    # Criterion: val improves AND test does not degrade beyond tolerance
    val_improves = new_val_auc >= prod_val_auc
    test_degrade = (prod_test_auc - new_test_auc) / (prod_test_auc or 1e-9)
    test_ok = test_degrade <= degrade_tol

    if val_improves and test_ok:
        logger.info(
            "Promotion criterion met (val: %.4f>=%.4f, test degradation %.2f%% <= %.2f%%); promoting %s",
            new_val_auc,
            prod_val_auc,
            test_degrade * 100,
            degrade_tol * 100,
            model_version,
        )
        _promote_model(root, config, model_version, new_metrics)
        return True

    reasons = []
    if not val_improves:
        reasons.append(f"val ROC-AUC {new_val_auc:.4f} < {prod_val_auc:.4f}")
    if not test_ok:
        reasons.append(f"test degradation {test_degrade*100:.2f}% > {degrade_tol*100:.2f}%")
    logger.info(
        "Retrain run not promoted: %s. Keeping current production %s",
        "; ".join(reasons),
        registry.get("current_production"),
    )
    return False


def _promote_model(root: Path, config: dict, model_version: str, new_metrics: dict) -> None:
    """Update registry to set model_version as current production."""
    from datetime import datetime

    registry_path = root / "model_registry.json"
    registry = _get_registry()
    models_dir = root / config["paths"]["models"]
    model_dir = models_dir / model_version

    version_entry = {
        "version": model_version,
        "path": str(model_dir.relative_to(root)),
        "val_roc_auc": new_metrics["validation"]["roc_auc"],
        "test_roc_auc": new_metrics["test"]["roc_auc"],
        "val_f1": new_metrics["validation"]["f1"],
        "test_f1": new_metrics["test"]["f1"],
        "trained_at": datetime.now().isoformat(),
    }
    if model_version not in [v.get("version") for v in registry.get("versions", [])]:
        registry.setdefault("versions", []).append(version_entry)
    registry["current_production"] = model_version
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated: current_production = %s", model_version)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run retraining pipeline (Section 12)")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip data ingestion (use existing raw)")
    parser.add_argument("--force-promotion", action="store_true", help="Promote new model regardless of criterion")
    parser.add_argument("--check-only", action="store_true", help="Only check if retrain is needed, then exit")
    args = parser.parse_args()

    if args.check_only:
        needed, reason = check_retrain_needed()
        print(f"Retrain needed: {needed}")
        print(f"Reason: {reason}")
        exit(0 if not needed else 1)

    promoted = run_retrain(skip_ingestion=args.skip_ingestion, force_promotion=args.force_promotion)
    exit(0 if promoted else 1)
