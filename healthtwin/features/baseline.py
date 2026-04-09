"""
HealthTwin — Personal Baseline Normalizer
===========================================
Implements personal baseline normalization — the key concept that makes
predictions PERSONALIZED rather than population-level.

Each user's metrics are normalized against their OWN historical baseline,
not population averages.  This enables detection of deviations that are
meaningful *for that individual*.

Usage::

    normalizer = PersonalBaselineNormalizer()
    normalizer.fit(df, user_id_col="user_id")
    normalized_df = normalizer.transform(df)
    profile = normalizer.get_user_profile("user_001")
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger


class PersonalBaselineNormalizer:
    """
    Per-user normalization against each individual's own baseline period.

    The first ``baseline_days`` of each user's data are used to establish
    their personal baseline.  All subsequent values are expressed as
    z-scores relative to that personal baseline.
    """

    def __init__(self, baseline_days: int = 30, anomaly_threshold: float = 2.0):
        """
        Parameters
        ----------
        baseline_days : int
            Number of days used as the personal baseline calibration period.
        anomaly_threshold : float
            Absolute z-score above which a value is flagged as anomalous.
        """
        self.baseline_days = baseline_days
        self.anomaly_threshold = anomaly_threshold
        self.baselines: dict[str, dict[str, dict[str, Any]]] = {}
        self._numeric_cols: list[str] = []

    # ------------------------------------------------------------------
    # fit() — learn each user's personal baseline
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, user_id_col: str = "user_id") -> None:
        """
        Compute personal baselines for every user in *df*.

        For each user and each numeric column, the first
        ``baseline_days`` of data are used to compute:

        - mean and std
        - percentiles at [5, 25, 50, 75, 95]
        - personal resting heart rate (5th percentile of HR during sleep)
        - personal max heart rate (95th percentile during activity)

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``timestamp`` and *user_id_col* columns plus
            numeric signal columns.
        user_id_col : str
            Column identifying individual users.
        """
        logger.info(
            f"Fitting personal baselines ({self.baseline_days}-day window) "
            f"for {df[user_id_col].nunique()} users"
        )

        skip_cols = {"timestamp", user_id_col, "user_id"}
        self._numeric_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in skip_cols
        ]

        for uid, udf in df.groupby(user_id_col):
            uid = str(uid)
            udf = udf.sort_values("timestamp").reset_index(drop=True)

            # Determine baseline period
            t_start = udf["timestamp"].iloc[0]
            t_end = t_start + pd.Timedelta(days=self.baseline_days)
            baseline = udf[udf["timestamp"] <= t_end]

            if len(baseline) < 96 * 7:  # fewer than ~7 days of 15-min data
                logger.warning(
                    f"  [{uid}] Only {len(baseline)} rows in baseline period "
                    f"(< 7 days) — results may be unreliable"
                )

            user_stats: dict[str, dict[str, Any]] = {}
            for col in self._numeric_cols:
                if col not in baseline.columns:
                    continue
                vals = baseline[col].dropna()
                if len(vals) < 2:
                    continue
                user_stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "percentiles": {
                        p: float(np.percentile(vals, p))
                        for p in [5, 25, 50, 75, 95]
                    },
                }

            # Special metrics if heart_rate and sleep_hours exist
            if "heart_rate" in baseline.columns and "sleep_hours" in baseline.columns:
                sleep_mask = baseline["sleep_hours"] > 0
                hr_during_sleep = baseline.loc[sleep_mask, "heart_rate"].dropna()
                hr_during_active = baseline.loc[~sleep_mask, "heart_rate"].dropna()

                user_stats["_personal_resting_hr"] = {
                    "value": float(
                        np.percentile(hr_during_sleep, 5)
                        if len(hr_during_sleep) > 0
                        else np.nan
                    )
                }
                user_stats["_personal_max_hr"] = {
                    "value": float(
                        np.percentile(hr_during_active, 95)
                        if len(hr_during_active) > 0
                        else np.nan
                    )
                }

            self.baselines[uid] = user_stats
            logger.debug(
                f"  [{uid}] Baseline fitted: {len(user_stats)} metrics, "
                f"{len(baseline)} rows"
            )

        logger.success(
            f"Personal baselines fitted for {len(self.baselines)} users"
        )

    # ------------------------------------------------------------------
    # transform() — normalise against personal baseline
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add personal z-scores, percentile ranks, and anomaly flags.

        New columns per numeric column:
        - ``{col}_personal_zscore``
        - ``{col}_personal_percentile``
        - ``{col}_deviation_flag`` (1 if |z| > threshold)

        Plus one aggregate column:
        - ``overall_anomaly_score`` = fraction of signals flagged

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``user_id`` column matched to fitted baselines.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with additional normalised columns.
        """
        result = df.copy()
        deviation_cols: list[str] = []

        for uid, mask in result.groupby("user_id").groups.items():
            uid_str = str(uid)
            if uid_str not in self.baselines:
                logger.warning(f"  [{uid_str}] No baseline found — skipping")
                continue

            stats = self.baselines[uid_str]
            idx = mask

            for col in self._numeric_cols:
                if col not in result.columns or col not in stats:
                    continue

                s = stats[col]
                vals = result.loc[idx, col]
                mean, std = s["mean"], s["std"]

                # Personal z-score
                zscore_col = f"{col}_personal_zscore"
                result.loc[idx, zscore_col] = (
                    (vals - mean) / std if std > 0 else 0.0
                )

                # Personal percentile rank (within user's baseline distribution)
                pctl_col = f"{col}_personal_percentile"
                pctls = s["percentiles"]
                result.loc[idx, pctl_col] = vals.apply(
                    lambda v: self._percentile_rank(v, pctls)
                )

                # Deviation flag
                flag_col = f"{col}_deviation_flag"
                result.loc[idx, flag_col] = (
                    result.loc[idx, zscore_col].abs() > self.anomaly_threshold
                ).astype(int)

                if flag_col not in deviation_cols:
                    deviation_cols.append(flag_col)

        # Overall anomaly score = fraction of monitored signals flagged
        if deviation_cols:
            existing = [c for c in deviation_cols if c in result.columns]
            if existing:
                result["overall_anomaly_score"] = (
                    result[existing].sum(axis=1) / len(existing)
                )

        logger.success(
            f"Personal normalisation applied: "
            f"+{len(deviation_cols) * 3 + 1} columns"
        )
        return result

    @staticmethod
    def _percentile_rank(value: float, percentiles: dict[int, float]) -> float:
        """Map a value to a 0–100 percentile rank using baseline percentiles."""
        if np.isnan(value):
            return np.nan
        p_values = sorted(percentiles.items())
        for i, (p, threshold) in enumerate(p_values):
            if value <= threshold:
                if i == 0:
                    return float(p)
                prev_p, prev_t = p_values[i - 1]
                # Linear interpolation between percentile brackets
                frac = (value - prev_t) / (threshold - prev_t + 1e-9)
                return prev_p + frac * (p - prev_p)
        return 95.0  # above 95th percentile

    # ------------------------------------------------------------------
    # get_user_profile() — human-readable health profile
    # ------------------------------------------------------------------

    def get_user_profile(self, user_id: str) -> dict:
        """
        Return a human-readable health profile for a fitted user.

        Returns
        -------
        dict
            Keys: user_id, baseline_period_days, resting_hr, typical_hrv,
            typical_sleep_hours, typical_stress_level, anomaly_history.
        """
        if user_id not in self.baselines:
            return {"error": f"No baseline found for user '{user_id}'"}

        stats = self.baselines[user_id]

        def _get_median(metric: str) -> float:
            if metric in stats and "percentiles" in stats[metric]:
                return stats[metric]["percentiles"].get(50, np.nan)
            return np.nan

        resting_hr = (
            stats.get("_personal_resting_hr", {}).get("value", np.nan)
        )

        return {
            "user_id": user_id,
            "baseline_period_days": self.baseline_days,
            "resting_hr": round(resting_hr, 1) if not np.isnan(resting_hr) else None,
            "typical_hrv": round(_get_median("hrv_rmssd"), 1),
            "typical_sleep_hours": round(_get_median("sleep_hours"), 1),
            "typical_heart_rate": round(_get_median("heart_rate"), 1),
            "typical_temp": round(_get_median("temp"), 2),
            "typical_resp_rate": round(_get_median("resp_rate"), 1),
            "anomaly_threshold": self.anomaly_threshold,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted normaliser to disk (pickle + YAML metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        # Human-readable metadata sidecar
        meta = {
            "baseline_days": self.baseline_days,
            "anomaly_threshold": self.anomaly_threshold,
            "users_fitted": list(self.baselines.keys()),
            "numeric_columns": self._numeric_cols,
        }
        meta_path = path.with_suffix(".meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

        logger.info(f"Saved normaliser to {path} + {meta_path}")

    @classmethod
    def load(cls, path: str) -> "PersonalBaselineNormalizer":
        """Load a previously saved normaliser from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        logger.info(f"Loaded normaliser from {path}")
        return obj


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)
    n = 4000  # ~42 days at 15-min intervals

    timestamps = pd.date_range("2023-06-01", periods=n, freq="15min")
    hours = timestamps.hour + timestamps.minute / 60
    circadian = np.sin(2 * np.pi * (hours - 6) / 24)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "user_id": "user_001",
        "heart_rate": 68 + 8 * circadian + np.random.normal(0, 3, n),
        "hrv_rmssd": 50 - 10 * circadian + np.random.normal(0, 5, n),
        "sleep_hours": np.where(
            (hours >= 22) | (hours < 7),
            np.clip(np.random.normal(7, 0.5, n), 0, 10),
            0,
        ),
        "temp": 36.6 + 0.3 * circadian + np.random.normal(0, 0.1, n),
        "resp_rate": 16 + 2 * circadian + np.random.normal(0, 1, n),
    })

    norm = PersonalBaselineNormalizer(baseline_days=14)
    norm.fit(df)

    result = norm.transform(df)
    profile = norm.get_user_profile("user_001")

    logger.info(f"Transformed shape: {result.shape}")
    logger.info(f"User profile: {profile}")

    # Verify z-scores are roughly standard normal after baseline period
    baseline_end = 14 * 96  # 14 days in 15-min rows
    post_baseline = result.iloc[baseline_end:]
    for col in ["heart_rate", "hrv_rmssd"]:
        zs = post_baseline[f"{col}_personal_zscore"].dropna()
        logger.info(
            f"  {col} z-score — mean: {zs.mean():.3f}, std: {zs.std():.3f}"
        )
