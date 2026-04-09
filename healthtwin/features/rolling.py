"""
HealthTwin — Rolling Time-Series Feature Engine
=================================================
Core of the "digital twin" concept: transforms raw time-series data into
rich rolling statistical features that capture trends, volatility, and
momentum across multiple time horizons (24h, 7d, 30d).

Usage::

    engine = RollingFeatureEngine()
    features_df = engine.fit_transform(df, user_id="demo_user")
    print(engine.get_feature_names())
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class RollingFeatureEngine:
    """
    Compute rolling statistical features over configurable time windows.

    Attributes
    ----------
    windows : list[int]
        Window sizes in number of rows (assumes 15-min intervals).
        Default: [96, 672, 2880] → 24 h, 7 d, 30 d.
    target_columns : list[str]
        Signal columns to compute features for.
    """

    # 24h=96, 7d=672, 30d=2880  (at 15-min granularity)
    DEFAULT_WINDOWS = [96, 672, 2880]
    DEFAULT_TARGETS = [
        "heart_rate",
        "hrv_rmssd",
        "eda_mean",
        "resp_rate",
        "temp",
        "activity_count",
        "sleep_hours",
    ]

    def __init__(
        self,
        windows: Optional[list[int]] = None,
        target_columns: Optional[list[str]] = None,
        min_periods: int = 10,
    ) -> None:
        self.windows = windows or self.DEFAULT_WINDOWS
        self.target_columns = target_columns or self.DEFAULT_TARGETS
        self.min_periods = min_periods
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, user_id: str = "demo") -> pd.DataFrame:
        """
        Generate rolling features for every target column × window size.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame sorted by timestamp with signal columns present.
        user_id : str
            User identifier (for logging only).

        Returns
        -------
        pd.DataFrame
            Original columns + all generated rolling / lag / cross-signal
            feature columns.
        """
        logger.info(
            f"[{user_id}] Building rolling features: "
            f"{len(self.windows)} windows × "
            f"{len(self._present_cols(df))} signals"
        )

        result = df.copy()
        present = self._present_cols(result)

        # --- Per-column rolling stats -----------------------------------
        for col in present:
            self._rolling_stats(result, col)
            self._lag_features(result, col)

        # --- Cross-signal interaction features --------------------------
        self._cross_signal_features(result)

        # Cache generated feature names
        new_cols = [c for c in result.columns if c not in df.columns]
        self._feature_names = sorted(new_cols)

        logger.success(
            f"[{user_id}] Generated {len(self._feature_names)} rolling features "
            f"({result.shape[0]} rows)"
        )
        return result

    def get_feature_names(self) -> list[str]:
        """Return sorted list of all generated feature column names."""
        return list(self._feature_names)

    def get_feature_report(self, df: pd.DataFrame) -> dict:
        """
        Diagnostic report on the generated feature matrix.

        Returns
        -------
        dict
            Keys: total_features, nan_percentage_per_feature (top 20 worst),
            feature_correlation_with_target (if 'risk_label' present).
        """
        feat_cols = [c for c in self._feature_names if c in df.columns]
        nan_pct = (df[feat_cols].isna().sum() / len(df) * 100).sort_values(
            ascending=False
        )

        report: dict = {
            "total_features": len(feat_cols),
            "nan_percentage_per_feature": nan_pct.head(20).to_dict(),
        }

        if "risk_label" in df.columns:
            corr = (
                df[feat_cols]
                .corrwith(df["risk_label"])
                .abs()
                .sort_values(ascending=False)
            )
            report["feature_correlation_with_target"] = corr.head(20).to_dict()

        return report

    # ------------------------------------------------------------------
    # Internal helpers (vectorised — no Python row loops)
    # ------------------------------------------------------------------

    def _present_cols(self, df: pd.DataFrame) -> list[str]:
        """Return target columns that actually exist in *df*."""
        return [c for c in self.target_columns if c in df.columns]

    def _rolling_stats(self, df: pd.DataFrame, col: str) -> None:
        """
        Add rolling mean, std, min, max, z-score, slope, pct_change
        for *col* across every configured window.
        """
        mp = self.min_periods

        for w in self.windows:
            prefix = f"{col}_{w}w"
            rolling = df[col].rolling(window=w, min_periods=mp)

            # Core statistics
            r_mean = rolling.mean()
            r_std = rolling.std()
            df[f"{prefix}_mean"] = r_mean
            df[f"{prefix}_std"] = r_std
            df[f"{prefix}_min"] = rolling.min()
            df[f"{prefix}_max"] = rolling.max()

            # Z-score: how far current value deviates from rolling mean
            df[f"{prefix}_zscore"] = (df[col] - r_mean) / r_std.replace(0, np.nan)

            # Percent change from rolling mean
            df[f"{prefix}_pct_change"] = (
                (df[col] - r_mean) / r_mean.replace(0, np.nan) * 100
            )

            # Linear trend slope (polyfit degree-1 over window)
            # Use cython engine for reliability across environments
            df[f"{prefix}_slope"] = (
                df[col]
                .rolling(window=w, min_periods=max(mp, 3))
                .apply(self._slope, raw=True)
            )

    @staticmethod
    def _slope(values: np.ndarray) -> float:
        """Compute linear trend slope via least-squares fit."""
        try:
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            return coeffs[0]
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

    def _lag_features(self, df: pd.DataFrame, col: str) -> None:
        """Add lag-1, lag-3, lag-7 features (in day-equivalent rows)."""
        # At 15-min intervals: 1 day = 96 rows, 3 days = 288, 7 days = 672
        for lag_days, lag_rows in [(1, 96), (3, 288), (7, 672)]:
            df[f"{col}_lag_{lag_days}d"] = df[col].shift(lag_rows)

    def _cross_signal_features(self, df: pd.DataFrame) -> None:
        """
        Compute interaction features across signals.

        Only generated when constituent columns are present.
        """
        if "sleep_hours" in df.columns and "heart_rate" in df.columns:
            df["sleep_hr_interaction"] = df["sleep_hours"] * df["heart_rate"]
            logger.debug("  + sleep_hr_interaction")

        if "eda_mean" in df.columns and "activity_count" in df.columns:
            df["stress_activity_ratio"] = df["eda_mean"] / (
                df["activity_count"] + 1
            )
            logger.debug("  + stress_activity_ratio")

        if "hrv_rmssd" in df.columns and "heart_rate" in df.columns:
            df["recovery_score"] = (
                df["hrv_rmssd"] / (df["heart_rate"] + 1) * 100
            )
            logger.debug("  + recovery_score")


# ======================================================================
# Self-test with synthetic data
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)
    n_rows = 2000  # ~20 days at 15-min intervals

    logger.info("Generating synthetic test data …")
    timestamps = pd.date_range("2023-06-01", periods=n_rows, freq="15min")

    # Simulate realistic circadian patterns
    hours = np.array(timestamps.hour + timestamps.minute / 60, dtype=float)
    circadian = np.sin(2 * np.pi * (hours - 6) / 24)  # peaks at noon

    synthetic = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": 70 + 10 * circadian + np.random.normal(0, 3, n_rows),
        "hrv_rmssd": 45 - 8 * circadian + np.random.normal(0, 5, n_rows),
        "eda_mean": 2.0 + 0.5 * circadian + np.random.normal(0, 0.3, n_rows),
        "resp_rate": 16 + 2 * circadian + np.random.normal(0, 1, n_rows),
        "temp": 36.5 + 0.3 * circadian + np.random.normal(0, 0.1, n_rows),
        "activity_count": np.maximum(
            0, 3000 + 2000 * circadian + np.random.normal(0, 500, n_rows)
        ),
        "sleep_hours": np.where(
            (hours >= 22) | (hours < 7),
            np.clip(np.random.normal(7.5, 0.5, n_rows), 4, 10),
            0,
        ),
    })

    engine = RollingFeatureEngine()
    result = engine.fit_transform(synthetic, user_id="test_user")

    logger.info(f"Output shape: {result.shape}")
    logger.info(f"Feature count: {len(engine.get_feature_names())}")
    logger.info(f"Sample features: {engine.get_feature_names()[:10]}")

    report = engine.get_feature_report(result)
    logger.info(f"Report: {report['total_features']} features total")
    logger.info("Top NaN%:")
    for k, v in list(report["nan_percentage_per_feature"].items())[:5]:
        logger.info(f"  {k}: {v:.1f}%")
