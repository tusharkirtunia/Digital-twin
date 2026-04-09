"""
HealthTwin — Master Feature Pipeline
======================================
Orchestrates the full data → features workflow:

    raw datasets  →  loaders  →  HRV processing  →  alignment
                   →  rolling features  →  baseline normalisation
                   →  final feature matrix

Usage::

    from features.pipeline import build_feature_matrix, get_demo_user_df

    # From real datasets
    feature_df, normalizer = build_feature_matrix(raw_dfs, user_id="s01")

    # Quick demo (synthetic / single-subject)
    demo_df = get_demo_user_df()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from features.rolling import RollingFeatureEngine
from features.baseline import PersonalBaselineNormalizer
from features.hrv import HRVFeatureExtractor


# ===================================================================
# Master pipeline
# ===================================================================

def build_feature_matrix(
    raw_dfs: dict[str, pd.DataFrame],
    user_id: str = "demo_user",
    nan_threshold: float = 0.60,
) -> tuple[pd.DataFrame, PersonalBaselineNormalizer]:
    """
    Build the complete feature matrix from one or more raw datasets.

    Parameters
    ----------
    raw_dfs : dict[str, pd.DataFrame]
        Optional keys: ``'heart'``, ``'pamap2'``, ``'wesad'``, ``'sleep'``.
        Pass whichever datasets are available.
    user_id : str
        User identifier.
    nan_threshold : float
        Drop columns with NaN fraction exceeding this value.

    Returns
    -------
    tuple[pd.DataFrame, PersonalBaselineNormalizer]
        The fully-featured DataFrame and the fitted normaliser.
    """
    logger.info(f"Building feature matrix for user '{user_id}'")
    logger.info(f"  Input datasets: {list(raw_dfs.keys())}")

    frames: list[pd.DataFrame] = []

    # ----- Process each dataset into a common schema ------------------

    if "heart" in raw_dfs:
        heart_df = raw_dfs["heart"].copy()
        heart_df["user_id"] = user_id
        frames.append(heart_df)
        logger.debug(f"  heart: {heart_df.shape}")

    if "pamap2" in raw_dfs:
        pamap = raw_dfs["pamap2"].copy()
        # Derive activity_count and heart_rate
        pamap["user_id"] = user_id
        # Compute accelerometer magnitude as activity proxy
        acc_cols = [c for c in pamap.columns if "acc" in c]
        if acc_cols:
            pamap["activity_count"] = np.sqrt(
                (pamap[acc_cols] ** 2).sum(axis=1)
            ) * 100  # scale to step-like units
        frames.append(pamap)
        logger.debug(f"  pamap2: {pamap.shape}")

    if "wesad" in raw_dfs:
        wesad = raw_dfs["wesad"].copy()
        wesad["user_id"] = user_id

        # Extract HRV features from ECG
        hrv_ext = HRVFeatureExtractor()
        try:
            wesad = hrv_ext.batch_process_wesad(
                wesad,
                ecg_sampling_rate=10,  # post-downsample rate
                window_seconds=300,
            )
            logger.debug(f"  wesad (with HRV): {wesad.shape}")
        except Exception as e:
            logger.warning(f"  HRV extraction failed: {e}")

        # Rename columns for pipeline consistency
        col_map = {
            "eda": "eda_mean",
            "resp": "resp_rate",
            "HRV_RMSSD": "hrv_rmssd",
        }
        wesad.rename(columns={k: v for k, v in col_map.items() if k in wesad.columns},
                     inplace=True)
        frames.append(wesad)

    if "sleep" in raw_dfs:
        sleep = raw_dfs["sleep"].copy()
        sleep["user_id"] = user_id

        # Convert sleep stages to sleep_hours proxy:
        # if stage > 0, user is sleeping → accumulate
        # (simplified for pipeline — real logic would use epoch durations)
        if "sleep_stage" in sleep.columns:
            sleep["is_sleeping"] = (sleep["sleep_stage"] > 0).astype(float)
            # Rolling 8-hour window to compute hours slept
            sleep["sleep_hours"] = (
                sleep["is_sleeping"]
                .rolling(window=960, min_periods=1)  # 960 × 30s = 8h
                .sum()
                / 120  # 120 epochs per hour (30s epochs)
            )
        frames.append(sleep)
        logger.debug(f"  sleep: {sleep.shape}")

    if not frames:
        raise ValueError("No datasets provided — pass at least one in raw_dfs")

    # ----- Merge/align to common 15-minute timestamps ------------------

    if len(frames) == 1:
        merged = frames[0].copy()
    else:
        # Resample each frame to 15-min and merge by timestamp
        resampled: list[pd.DataFrame] = []
        for i, df in enumerate(frames):
            df = df.copy()
            if "timestamp" not in df.columns:
                continue
            df = df.set_index("timestamp")
            numeric = df.select_dtypes(include=[np.number])
            r = numeric.resample("15min").mean()
            r["user_id"] = user_id
            resampled.append(r.reset_index())

        merged = resampled[0]
        for r in resampled[1:]:
            # Drop overlapping columns (except timestamp, user_id)
            overlap_cols = set(merged.columns) & set(r.columns) - {"timestamp", "user_id"}
            r_trimmed = r.drop(columns=list(overlap_cols), errors="ignore")
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                r_trimmed.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("30min"),
            )

    logger.info(f"  Aligned shape: {merged.shape}")

    # ----- Apply Rolling Feature Engine --------------------------------

    engine = RollingFeatureEngine()
    featured = engine.fit_transform(merged, user_id=user_id)

    # ----- Apply Personal Baseline Normalisation -----------------------

    normalizer = PersonalBaselineNormalizer(baseline_days=14)
    normalizer.fit(featured)
    featured = normalizer.transform(featured)

    # ----- Drop high-NaN columns ----------------------------------------

    nan_frac = featured.isna().sum() / len(featured)
    drop_cols = nan_frac[nan_frac > nan_threshold].index.tolist()
    # Never drop core identifiers
    drop_cols = [c for c in drop_cols if c not in ("timestamp", "user_id")]
    if drop_cols:
        logger.info(
            f"  Dropping {len(drop_cols)} high-NaN columns "
            f"(>{nan_threshold*100:.0f}%)"
        )
        featured.drop(columns=drop_cols, inplace=True)

    logger.success(
        f"Feature matrix ready: {featured.shape[0]} rows × "
        f"{featured.shape[1]} columns"
    )
    return featured, normalizer


# ===================================================================
# Quick demo user generator
# ===================================================================

def get_demo_user_df() -> pd.DataFrame:
    """
    Generate a ready-to-model DataFrame from synthetic data.

    This is the fast-path for testing without downloading any real
    datasets.  Generates 30 days of realistic synthetic wearable data,
    runs the full feature pipeline, and returns the result.

    Returns
    -------
    pd.DataFrame
        Fully-featured DataFrame ready for model training or dashboard.
    """
    np.random.seed(42)
    n = 2880  # 30 days × 96 rows/day (15-min intervals)
    timestamps = pd.date_range("2023-06-01", periods=n, freq="15min")
    hours = np.array(timestamps.hour + timestamps.minute / 60, dtype=float)

    # Realistic circadian patterns — convert to numpy to avoid Index issues
    circadian = np.sin(2 * np.pi * (hours - 6) / 24)
    day_of_week = np.array(timestamps.dayofweek, dtype=float)
    is_weekend = (day_of_week >= 5).astype(float)

    # Synthesise a multi-signal wearable stream
    synthetic = pd.DataFrame({
        "timestamp": timestamps,
        "user_id": "demo_user",
        "heart_rate": (
            68 + 10 * circadian
            - 3 * is_weekend  # lower HR on weekends
            + np.random.normal(0, 3, n)
        ),
        "hrv_rmssd": np.clip(
            48 - 8 * circadian
            + 5 * is_weekend
            + np.random.normal(0, 5, n),
            10, 120,
        ),
        "eda_mean": np.clip(
            1.5 + 0.8 * circadian
            - 0.3 * is_weekend
            + np.random.normal(0, 0.3, n),
            0.1, 10,
        ),
        "resp_rate": np.clip(
            15 + 2 * circadian
            + np.random.normal(0, 1, n),
            8, 30,
        ),
        "temp": (
            36.5 + 0.3 * circadian
            + np.random.normal(0, 0.1, n)
        ),
        "activity_count": np.maximum(0, (
            3500 + 3000 * circadian
            + 1000 * is_weekend
            + np.random.normal(0, 800, n)
        )),
        "sleep_hours": np.where(
            (hours >= 22) | (hours < 7),
            np.clip(np.random.normal(7.2, 0.6, n), 4, 10),
            0.0,
        ),
    })

    # Add a subtle deterioration trend in last 10 days for realism
    deterioration_start = n - 960  # last 10 days
    synthetic.loc[deterioration_start:, "heart_rate"] += np.linspace(0, 8, n - deterioration_start)
    synthetic.loc[deterioration_start:, "hrv_rmssd"] -= np.linspace(0, 12, n - deterioration_start)
    synthetic.loc[deterioration_start:, "sleep_hours"] *= np.linspace(1.0, 0.75, n - deterioration_start)

    logger.info(f"Generated synthetic demo data: {synthetic.shape}")

    # Run through the pipeline (rolling + baseline only — no HRV extraction)
    engine = RollingFeatureEngine()
    featured = engine.fit_transform(synthetic, user_id="demo_user")

    normalizer = PersonalBaselineNormalizer(baseline_days=14)
    normalizer.fit(featured)
    featured = normalizer.transform(featured)

    # Drop extreme NaN columns
    nan_frac = featured.isna().sum() / len(featured)
    drop_cols = nan_frac[nan_frac > 0.6].index.tolist()
    drop_cols = [c for c in drop_cols if c not in ("timestamp", "user_id")]
    featured.drop(columns=drop_cols, inplace=True, errors="ignore")

    logger.success(
        f"Demo feature matrix: {featured.shape[0]} rows × "
        f"{featured.shape[1]} columns"
    )
    return featured


# ===================================================================
# CLI entrypoint
# ===================================================================

if __name__ == "__main__":
    logger.info("Running feature pipeline demo …")
    df = get_demo_user_df()

    logger.info(f"\nFinal shape: {df.shape}")
    logger.info(f"Columns ({len(df.columns)}):")
    for i, col in enumerate(sorted(df.columns)):
        logger.info(f"  {i+1:3d}. {col}")

    # Sample of feature values from the middle of the dataset
    mid = len(df) // 2
    sample = df.iloc[mid]
    logger.info("\nSample row (middle of dataset):")
    for col in ["heart_rate", "heart_rate_96w_mean", "heart_rate_96w_zscore",
                 "heart_rate_personal_zscore", "recovery_score",
                 "overall_anomaly_score"]:
        if col in df.columns:
            logger.info(f"  {col}: {sample[col]:.3f}")
