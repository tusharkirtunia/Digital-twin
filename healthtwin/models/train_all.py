"""
HealthTwin — Model Training Script
====================================
Train all three health models in sequence and save to disk.

Usage::

    python -m models.train_all --models all --verbose
    python -m models.train_all --models cardiac stress --output-path models/saved
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.pipeline import get_demo_user_df
from models.risk_model import CardiacRiskModel, StressModel, SleepQualityModel


def train_all(
    models_to_train: list[str],
    data_path: str = "data/raw",
    output_path: str = "models/saved",
    verbose: bool = False,
) -> dict[str, dict]:
    """
    Train selected models and save to disk.

    Parameters
    ----------
    models_to_train : list[str]
        Which models: ``['cardiac', 'stress', 'sleep', 'all']``.
    data_path : str
        Root path for raw dataset files.
    output_path : str
        Directory to save trained model artifacts.
    verbose : bool
        Enable debug-level logging.

    Returns
    -------
    dict[str, dict]
        Metrics per model.
    """
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "all" in models_to_train:
        models_to_train = ["cardiac", "stress", "sleep"]

    logger.info("=" * 60)
    logger.info("HealthTwin — Model Training Pipeline")
    logger.info(f"  Models: {models_to_train}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Timestamp: {timestamp}")
    logger.info("=" * 60)

    # ----- Load data -------------------------------------------------------
    # Try real datasets first, fall back to demo synthetic data
    logger.info("Loading training data …")

    # Attempt UCI Heart Disease
    uci_path = Path(data_path) / "uci_heart" / "heart_cleveland.csv"
    uci_df = None
    if uci_path.exists():
        from data.processed.loaders import load_uci_heart
        uci_df = load_uci_heart(str(uci_path))
        logger.info(f"  UCI Heart Disease: {uci_df.shape}")
    else:
        logger.info("  UCI Heart Disease not found — using demo data")

    # Generate feature-rich demo data for stress/sleep models
    demo_df = get_demo_user_df()
    logger.info(f"  Demo data: {demo_df.shape}")

    # ----- Train models ---------------------------------------------------
    all_metrics: dict[str, dict] = {}

    # 1. Cardiac Risk Model
    if "cardiac" in models_to_train:
        logger.info("\n" + "─" * 40)
        logger.info("Training: Cardiac Risk Model")
        logger.info("─" * 40)

        cardiac = CardiacRiskModel()

        if uci_df is not None:
            metrics = cardiac.train(uci_df)
        else:
            # Use demo data with synthetic cardiac target
            df_cardiac = demo_df.copy()
            # Simulate cardiac risk: elevated HR + low HRV + low sleep
            cardiac_risk = pd.Series(0, index=df_cardiac.index)
            if "heart_rate" in df_cardiac.columns:
                cardiac_risk += (df_cardiac["heart_rate"] > 75).astype(int)
            if "hrv_rmssd" in df_cardiac.columns:
                cardiac_risk += (df_cardiac["hrv_rmssd"] < 35).astype(int)
            if "sleep_hours" in df_cardiac.columns:
                cardiac_risk += (df_cardiac["sleep_hours"] < 6).astype(int)
            df_cardiac["target"] = (cardiac_risk >= 2).astype(int)
            metrics = cardiac.train(df_cardiac)

        cardiac.save(str(output_dir / f"cardiac_{timestamp}.joblib"))
        # Also save as "latest" for easy loading
        cardiac.save(str(output_dir / "cardiac_latest.joblib"))
        all_metrics["cardiac"] = metrics

    # 2. Stress Model
    if "stress" in models_to_train:
        logger.info("\n" + "─" * 40)
        logger.info("Training: Stress Model")
        logger.info("─" * 40)

        stress = StressModel()
        metrics = stress.train(demo_df)

        stress.save(str(output_dir / f"stress_{timestamp}.joblib"))
        stress.save(str(output_dir / "stress_latest.joblib"))
        all_metrics["stress"] = metrics

    # 3. Sleep Quality Model
    if "sleep" in models_to_train:
        logger.info("\n" + "─" * 40)
        logger.info("Training: Sleep Quality Model")
        logger.info("─" * 40)

        sleep = SleepQualityModel()
        metrics = sleep.train(demo_df)

        sleep.save(str(output_dir / f"sleep_{timestamp}.joblib"))
        sleep.save(str(output_dir / "sleep_latest.joblib"))
        all_metrics["sleep"] = metrics

    # ----- Summary ---------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model':<12} {'AUC':>8} {'F1':>8} {'Acc':>8} {'Train':>8} {'Test':>8}")
    logger.info("─" * 60)
    for name, m in all_metrics.items():
        logger.info(
            f"{name:<12} {m['auc']:>8.3f} {m['f1']:>8.3f} "
            f"{m['accuracy']:>8.3f} {m['train_samples']:>8d} {m['test_samples']:>8d}"
        )
    logger.info("=" * 60)

    # Save metrics to YAML
    metrics_path = output_dir / "metrics.yaml"
    serializable = {}
    for k, v in all_metrics.items():
        serializable[k] = {
            mk: mv for mk, mv in v.items()
            if not isinstance(mv, (np.ndarray, np.generic))
        }
    with open(metrics_path, "w") as f:
        yaml.dump(serializable, f, default_flow_style=False)
    logger.info(f"\nMetrics saved to {metrics_path}")

    return all_metrics


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HealthTwin — Train all health risk models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["cardiac", "stress", "sleep", "all"],
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--data-path",
        default="data/raw",
        help="Root path for raw datasets (default: data/raw)",
    )
    parser.add_argument(
        "--output-path",
        default="models/saved",
        help="Directory for saved models (default: models/saved)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    train_all(
        models_to_train=args.models,
        data_path=args.data_path,
        output_path=args.output_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
