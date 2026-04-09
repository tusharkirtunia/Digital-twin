"""
HealthTwin — Automated Pipeline Tests
=======================================
Basic sanity tests to ensure feature engineering and model scoring
don't regress during hackathon updates.

Run with:
    python -m pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from features.pipeline import get_demo_user_df
from features.rolling import RollingFeatureEngine
from features.baseline import PersonalBaselineNormalizer
from models.risk_model import CardiacRiskModel, StressModel, SleepQualityModel

def test_demo_data_generation():
    """Test that synthetic demo data generates valid shapes and core signals."""
    df = get_demo_user_df()
    
    assert not df.empty, "Demo dataframe should not be empty"
    # Should have 30 days of 15-min data = 2880 rows
    assert len(df) == 2880
    
    # Core signals must be present
    core_signals = ["heart_rate", "hrv_rmssd", "eda_mean", "sleep_hours"]
    for sig in core_signals:
        assert sig in df.columns, f"Missing core signal {sig}"
        

def test_rolling_engine():
    """Test the rolling feature engineer on a small dataframe."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="15min"),
        "heart_rate": np.linspace(60, 100, 100),
        "sleep_hours": np.ones(100) * 8.0
    })
    
    engine = RollingFeatureEngine(windows=[10, 20])
    out = engine.fit_transform(df, user_id="test")
    
    # Should generate mean, std, zscore, slope, lag etc.
    assert "heart_rate_10w_mean" in out.columns
    assert "heart_rate_10w_slope" in out.columns
    assert "sleep_hr_interaction" in out.columns


def test_baseline_normalization():
    """Test that the personal baseline normalizer scales correctly."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
        "user_id": "test_user",
        "heart_rate": np.random.normal(70, 5, 100)
    })
    
    norm = PersonalBaselineNormalizer(baseline_days=10) # 10 days since our freq is daily
    norm.fit(df)
    out = norm.transform(df)
    
    assert "heart_rate_personal_zscore" in out.columns
    assert "heart_rate_personal_percentile" in out.columns
    
    # Z-scores should roughly center around mean
    zscore_mean = out["heart_rate_personal_zscore"].dropna().mean()
    assert abs(zscore_mean) < 1.0


def test_model_training_and_scoring():
    """Test that models can train and output valid probability scores."""
    # Create simple mock data
    np.random.seed(42)
    df = pd.DataFrame({
        "target": np.random.randint(0, 2, 200),
        "heart_rate": np.random.normal(70, 10, 200),
        "hrv_rmssd": np.random.normal(40, 10, 200),
        "col1": np.ones(200)
    })
    
    model = CardiacRiskModel()
    metrics = model.train(df)
    
    assert "auc" in metrics
    assert model.is_trained
    
    # Test realtime scoring
    row = df.iloc[0]
    score = model.score_realtime(row)
    
    assert "risk_score" in score
    assert 0.0 <= score["risk_score"] <= 1.0
    assert "risk_level" in score
    assert "top_risk_factors" in score
    
    # Test missing columns fallback
    bad_row = pd.Series({"unknown_col": 100})
    bad_score = model.score_realtime(bad_row)
    assert bad_score["risk_score"] >= 0.0 # Should fallback gracefully filling 0s
