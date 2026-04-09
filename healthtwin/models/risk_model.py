"""
HealthTwin — Health Risk Models
================================
XGBoost-based models for cardiac risk, stress detection, and sleep quality.

All three models share a ``BaseHealthModel`` abstract interface:
  - train(df) → metrics dict
  - score_realtime(feature_row) → risk dict
  - score_timeseries(feature_df) → scored DataFrame
  - save(path) / load(path)

Usage::

    model = CardiacRiskModel()
    metrics = model.train(uci_df)
    risk = model.score_realtime(feature_row)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imbalanced-learn not available — SMOTE disabled")


# ===================================================================
# Abstract base
# ===================================================================

class BaseHealthModel(ABC):
    """Abstract interface shared by all HealthTwin risk models."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list[str] = []
        self.is_trained: bool = False

    @abstractmethod
    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train the model. Returns metrics dict."""

    @abstractmethod
    def score_realtime(self, feature_row: pd.Series) -> dict[str, Any]:
        """Score a single observation. Returns risk dict."""

    def score_timeseries(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Apply score_realtime across an entire DataFrame."""
        result = feature_df.copy()
        scores = []
        levels = []
        for _, row in result.iterrows():
            out = self.score_realtime(row)
            scores.append(out["risk_score"])
            levels.append(out["risk_level"])

        result["risk_score"] = scores
        result["risk_level"] = levels

        # 7-day rolling average (672 rows at 15-min intervals)
        result["risk_score_7d_avg"] = (
            result["risk_score"].rolling(window=672, min_periods=10).mean()
        )
        # Flag spikes: current > 130% of 7d average
        result["risk_spike"] = (
            result["risk_score"] > result["risk_score_7d_avg"] * 1.3
        ).astype(int)

        return result

    def save(self, path: str) -> None:
        """Serialize the full model to disk with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "name": self.name,
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }
        joblib.dump(state, path)
        logger.info(f"Saved {self.name} model to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseHealthModel":
        """Deserialize a saved model from disk."""
        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj.name = state["name"]
        obj.pipeline = state["pipeline"]
        obj.feature_names = state["feature_names"]
        obj.is_trained = state["is_trained"]
        logger.info(f"Loaded {obj.name} model from {path}")
        return obj

    def _align_features(self, row: pd.Series) -> np.ndarray:
        """
        Align a feature row to the exact training schema.

        Missing columns → 0.0, extra columns → dropped.
        Returns a 1-D numpy array in training column order.
        """
        aligned = np.zeros(len(self.feature_names))
        for i, fname in enumerate(self.feature_names):
            if fname in row.index:
                val = row[fname]
                aligned[i] = val if not pd.isna(val) else 0.0
        return aligned

    @staticmethod
    def _risk_level(score: float) -> str:
        """Map a 0-1 risk score to a human-readable level."""
        if score < 0.3:
            return "Low"
        elif score < 0.55:
            return "Moderate"
        elif score < 0.75:
            return "High"
        return "Critical"


# ===================================================================
# 1. Cardiac Risk Model (UCI Heart Disease backbone)
# ===================================================================

class CardiacRiskModel(BaseHealthModel):
    """
    XGBoost classifier for cardiac risk, trained on UCI Heart Disease data.

    Applied to rolling features from wearable data for real-time risk scoring.
    """

    def __init__(self) -> None:
        super().__init__("cardiac")

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Train on UCI Heart Disease DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a ``target`` column (0/1) and numeric features.

        Returns
        -------
        dict
            Metrics: auc, f1, precision, recall, confusion_matrix.
        """
        logger.info(f"Training {self.name} model on {len(df)} samples")

        # Separate features and target
        exclude = {"target", "user_id", "timestamp"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df["target"].values

        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        # Build pipeline: Imputer → Scaler → XGBoost
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        if HAS_IMBLEARN:
            self.pipeline = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("model", xgb),
            ])
        else:
            self.pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", xgb),
            ])

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": self.name,
            "auc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        logger.success(
            f"{self.name} — AUC: {metrics['auc']:.3f}, "
            f"F1: {metrics['f1']:.3f}, Acc: {metrics['accuracy']:.3f}"
        )
        return metrics

    def score_realtime(self, feature_row: pd.Series) -> dict[str, Any]:
        """
        Score a single observation for cardiac risk.

        Returns
        -------
        dict
            risk_score (0-1), risk_level, top_risk_factors (top 5 features).
        """
        if not self.is_trained:
            return {"risk_score": 0.5, "risk_level": "Unknown", "top_risk_factors": []}

        aligned = self._align_features(feature_row).reshape(1, -1)
        proba = self.pipeline.predict_proba(aligned)[0, 1]
        score = float(proba)

        # Get top contributing features by absolute magnitude
        # (proper SHAP is in the explainer module — this is a fast proxy)
        feature_importance = np.abs(aligned[0])
        top_idx = np.argsort(feature_importance)[::-1][:5]
        top_factors = [self.feature_names[i] for i in top_idx if i < len(self.feature_names)]

        return {
            "risk_score": score,
            "risk_level": self._risk_level(score),
            "top_risk_factors": top_factors,
        }


# ===================================================================
# 2. Stress Model (WESAD / HRV features backbone)
# ===================================================================

class StressModel(BaseHealthModel):
    """
    XGBoost classifier for stress detection using HRV + EDA features.

    Trained on WESAD data where label==2 → stress, else → non-stress.
    Can also train on synthetic data with stress_index as proxy target.
    """

    def __init__(self) -> None:
        super().__init__("stress")

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Train on a DataFrame with physiological features.

        Expects either:
        - ``label`` column from WESAD (2=stress, else=0)
        - ``stress_label`` binary column
        - Falls back to synthetic target from EDA/HRV thresholds
        """
        logger.info(f"Training {self.name} model on {len(df)} samples")

        # Determine target column
        if "label" in df.columns:
            df = df.copy()
            df["stress_target"] = (df["label"] == 2).astype(int)
        elif "stress_label" in df.columns:
            df = df.copy()
            df["stress_target"] = df["stress_label"].astype(int)
        else:
            # Synthesise stress labels from available signals
            df = df.copy()
            stress_cond = pd.Series(False, index=df.index)
            if "eda_mean" in df.columns:
                stress_cond |= df["eda_mean"] > df["eda_mean"].quantile(0.75)
            if "hrv_rmssd" in df.columns:
                stress_cond |= df["hrv_rmssd"] < df["hrv_rmssd"].quantile(0.25)
            if "heart_rate" in df.columns:
                stress_cond |= df["heart_rate"] > df["heart_rate"].quantile(0.80)
            df["stress_target"] = stress_cond.astype(int)
            logger.info("  Using synthetic stress labels from signal thresholds")

        exclude = {"stress_target", "label", "stress_label", "user_id",
                    "timestamp", "target", "activity_name", "activity_id"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df["stress_target"].values

        # Drop rows with NaN target
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", xgb),
        ])

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": self.name,
            "auc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        logger.success(
            f"{self.name} — AUC: {metrics['auc']:.3f}, "
            f"F1: {metrics['f1']:.3f}, Acc: {metrics['accuracy']:.3f}"
        )
        return metrics

    def score_realtime(self, feature_row: pd.Series) -> dict[str, Any]:
        """Score a single observation for stress level."""
        if not self.is_trained:
            return {"risk_score": 0.5, "risk_level": "Unknown", "top_risk_factors": []}

        aligned = self._align_features(feature_row).reshape(1, -1)
        proba = self.pipeline.predict_proba(aligned)[0, 1]
        score = float(proba)

        feature_importance = np.abs(aligned[0])
        top_idx = np.argsort(feature_importance)[::-1][:5]
        top_factors = [self.feature_names[i] for i in top_idx if i < len(self.feature_names)]

        return {
            "risk_score": score,
            "risk_level": self._risk_level(score),
            "top_risk_factors": top_factors,
        }


# ===================================================================
# 3. Sleep Quality Model
# ===================================================================

class SleepQualityModel(BaseHealthModel):
    """
    XGBoost classifier for sleep quality.

    Target: binary good_sleep (1 if deep+REM sleep proportion > 40%).
    Can also train on synthetic data using sleep_hours as proxy.
    """

    def __init__(self) -> None:
        super().__init__("sleep")

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Train on sleep data.

        Expects either:
        - ``sleep_stage`` column (0=Wake,1=N1,2=N2,3=N3,4=REM)
        - ``good_sleep`` binary column
        - Falls back to synthetic labels from sleep_hours threshold
        """
        logger.info(f"Training {self.name} model on {len(df)} samples")

        df = df.copy()

        if "sleep_stage" in df.columns:
            # Compute per-night sleep quality from staging
            # Good sleep = N3 + REM make up > 40% of total sleep
            df["is_deep_rem"] = df["sleep_stage"].isin([3, 4]).astype(float)
            df["is_sleep"] = (df["sleep_stage"] > 0).astype(float)

            # Rolling window per sleep session (8h)
            window = min(960, len(df))  # 480 × 30s = 4h minimum
            df["deep_rem_frac"] = (
                df["is_deep_rem"].rolling(window=window, min_periods=10).mean()
            )
            df["sleep_target"] = (df["deep_rem_frac"] > 0.40).astype(int)
        elif "good_sleep" in df.columns:
            df["sleep_target"] = df["good_sleep"].astype(int)
        elif "sleep_hours" in df.columns:
            # Synthetic target: good sleep if sleep_hours > 6.5
            df["sleep_target"] = (df["sleep_hours"] > 6.5).astype(int)
            logger.info("  Using synthetic sleep quality labels (hours > 6.5)")
        else:
            raise ValueError("No sleep target column found in training data")

        exclude = {"sleep_target", "good_sleep", "sleep_stage", "is_deep_rem",
                    "is_sleep", "deep_rem_frac", "user_id", "timestamp",
                    "target", "label", "activity_name", "activity_id"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df["sleep_target"].values

        # Drop NaN targets
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid].astype(int)

        # Ensure sufficient samples in each class
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logger.warning("  Only one class present — adding synthetic minority samples")
            minority = 1 - unique[0]
            n_add = max(50, int(len(y) * 0.1))
            X_noise = X[:n_add] + np.random.normal(0, 0.1, (n_add, X.shape[1]))
            y_noise = np.full(n_add, minority)
            X = np.vstack([X, X_noise])
            y = np.concatenate([y, y_noise])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", xgb),
        ])

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": self.name,
            "auc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        logger.success(
            f"{self.name} — AUC: {metrics['auc']:.3f}, "
            f"F1: {metrics['f1']:.3f}, Acc: {metrics['accuracy']:.3f}"
        )
        return metrics

    def score_realtime(self, feature_row: pd.Series) -> dict[str, Any]:
        """Score a single observation for sleep quality."""
        if not self.is_trained:
            return {"risk_score": 0.5, "risk_level": "Unknown", "top_risk_factors": []}

        aligned = self._align_features(feature_row).reshape(1, -1)
        # For sleep, we invert: high proba of "good sleep" = low risk
        proba_good = self.pipeline.predict_proba(aligned)[0, 1]
        score = float(1.0 - proba_good)  # risk = 1 - quality

        feature_importance = np.abs(aligned[0])
        top_idx = np.argsort(feature_importance)[::-1][:5]
        top_factors = [self.feature_names[i] for i in top_idx if i < len(self.feature_names)]

        return {
            "risk_score": score,
            "risk_level": self._risk_level(score),
            "top_risk_factors": top_factors,
        }


# ===================================================================
# Self-test
# ===================================================================

if __name__ == "__main__":
    from features.pipeline import get_demo_user_df

    logger.info("Testing risk models with demo data …")
    df = get_demo_user_df()

    # --- Cardiac (using demo data with synthetic target) ---
    df["target"] = (df["heart_rate"] > df["heart_rate"].median()).astype(int)
    cardiac = CardiacRiskModel()
    cm = cardiac.train(df)
    logger.info(f"Cardiac metrics: {cm}")

    # --- Stress ---
    stress = StressModel()
    sm = stress.train(df)
    logger.info(f"Stress metrics: {sm}")

    # --- Sleep ---
    sleep = SleepQualityModel()
    slm = sleep.train(df)
    logger.info(f"Sleep metrics: {slm}")

    # --- Real-time scoring test ---
    sample_row = df.iloc[len(df) // 2]
    for model, name in [(cardiac, "cardiac"), (stress, "stress"), (sleep, "sleep")]:
        result = model.score_realtime(sample_row)
        logger.info(f"  {name}: score={result['risk_score']:.3f} level={result['risk_level']}")

    logger.info("All model tests passed ✓")
