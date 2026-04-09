"""
HealthTwin — Simulated Real-Time Playback Engine
===================================================
Replays a historical dataset chronologically, feeding data point-by-point
into the models, simulating what a live wearable data feed would do.

Designed to be thread-safe for use in Streamlit background threads.

Usage::

    engine = RealtimePlaybackEngine(df, models_dict, speed_multiplier=10.0)
    for state in engine.stream():
        print(state["timestamp"], state["risk_scores"])
"""

from __future__ import annotations

import time
import threading
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
from loguru import logger

from features.rolling import RollingFeatureEngine
from features.baseline import PersonalBaselineNormalizer
from models.risk_model import BaseHealthModel


class RealtimePlaybackEngine:
    """
    Simulates a real-time data stream from historical data.

    Maintains an internal 30-day rolling window to compute features
    on-the-fly for the simulated "current" timestamp.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        models: dict[str, BaseHealthModel],
        normalizer: Optional[PersonalBaselineNormalizer] = None,
        speed_multiplier: float = 1.0,
        start_index: int = 0,
        update_interval_sec: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset, sorted by timestamp. Ideally just raw signals.
        models : dict[str, BaseHealthModel]
            Dict of initialized and trained risk models.
        normalizer : PersonalBaselineNormalizer
            Fitted baseline normalizer. If None, normalization is skipped.
        speed_multiplier : float
            1.0 = real-time, 10.0 = 10x faster, 0.0 = instant (no sleep).
        start_index : int
            Row index to start streaming from.
        update_interval_sec : float
            Base wall-clock seconds between yielded rows before 
            speed_multiplier is applied.
        """
        self.full_df = df.sort_values("timestamp").reset_index(drop=True)
        self.models = models
        self.normalizer = normalizer
        self.speed_multiplier = float(speed_multiplier)
        self.update_interval_sec = update_interval_sec
        
        self.current_index = start_index
        
        # Max window needed is 30 days = 2880 rows (at 15min)
        self.max_buffer_size = 2880
        
        # Internal state
        self._lock = threading.Lock()
        self._is_running = False
        self._latest_state: dict[str, Any] = {}
        self._feature_engine = RollingFeatureEngine()

    def reset(self, start_index: int = 0) -> None:
        """Reset playback to a specific row index."""
        with self._lock:
            self.current_index = max(0, min(start_index, len(self.full_df) - 1))
            self._latest_state = {}
            logger.info(f"Playback reset to index {self.current_index}")

    def get_current_state(self) -> dict[str, Any]:
        """Return the latest emitted state snapshot safely."""
        with self._lock:
            return self._latest_state.copy()

    def stream(self) -> Generator[dict[str, Any], None, None]:
        """
        Stream the dataset row by row.

        Yields
        ------
        dict
            Contains timestamp, raw_signals, rolling_features, risk_scores,
            anomaly_flags, and personal_zscores.
        """
        self._is_running = True
        logger.info(
            f"Starting stream from index {self.current_index} "
            f"at {self.speed_multiplier}x speed"
        )

        while self._is_running and self.current_index < len(self.full_df):
            # 1. Grab historical window up to current point
            start_idx = max(0, self.current_index - self.max_buffer_size)
            window_df = self.full_df.iloc[start_idx : self.current_index + 1].copy()
            
            # The "current" row being simulated
            raw_row = window_df.iloc[-1]
            timestamp = raw_row["timestamp"]

            # 2. Recompute features incrementally
            # (In a true production system, we'd only compute the final row.
            # Here we compute the window but only keep the last row for simplicity).
            feat_df = self._feature_engine.fit_transform(window_df, user_id="live")
            
            if self.normalizer:
                # Add user_id so normalizer knows which baseline to use
                if "user_id" not in feat_df.columns:
                    feat_df["user_id"] = raw_row.get("user_id", "demo_user")
                feat_df = self.normalizer.transform(feat_df)

            curr_feat_row = feat_df.iloc[-1]

            # 3. Model Scoring
            risk_scores = {}
            for name, model in self.models.items():
                risk_scores[name] = model.score_realtime(curr_feat_row)

            # 4. Package state
            # Extract distinct groups of features for the UI
            zscores = {
                k: v for k, v in curr_feat_row.items() 
                if isinstance(k, str) and k.endswith("_personal_zscore")
            }
            flags = {
                k: v for k, v in curr_feat_row.items() 
                if isinstance(k, str) and k.endswith("_deviation_flag")
            }
            
            # Extract raw signals dynamically (exclude heavy feature cols)
            raw_cols = [c for c in self.full_df.columns if c not in ('timestamp', 'user_id')]
            raw_signals = {k: curr_feat_row[k] for k in raw_cols if k in curr_feat_row}

            state = {
                "timestamp": timestamp,
                "raw_signals": raw_signals,
                "risk_scores": risk_scores,
                "personal_zscores": zscores,
                "anomaly_flags": flags,
                "overall_anomaly_score": curr_feat_row.get("overall_anomaly_score", 0.0),
                "full_feature_row": curr_feat_row.to_dict(), # Useful for debug/explainers
            }

            with self._lock:
                self._latest_state = state
                self.current_index += 1

            yield state

            # 5. Emulate real-time pacing
            if self.speed_multiplier > 0:
                time.sleep(self.update_interval_sec / self.speed_multiplier)

        self._is_running = False
        logger.info("Playback stream ended.")

    def stop(self) -> None:
        """Signal the stream to gracefully stop."""
        self._is_running = False

    def export_playback_session(self, output_path: str) -> None:
        """
        Run the full playback instantly and save state history to CSV.
        Useful for debugging exactly what the engine produces.
        """
        logger.info(f"Exporting playback session to {output_path} (instant speed)")
        orig_speed = self.speed_multiplier
        orig_idx = self.current_index
        
        self.speed_multiplier = 0.0
        self.reset(0)
        
        history = []
        for state in self.stream():
            row = {"timestamp": state["timestamp"]}
            # Flatten risk scores
            for k, v in state["risk_scores"].items():
                row[f"{k}_risk"] = v["risk_score"]
                row[f"{k}_level"] = v["risk_level"]
            row.update(state["raw_signals"])
            history.append(row)
            
        df_out = pd.DataFrame(history)
        df_out.to_csv(output_path, index=False)
        logger.success(f"Exported {len(df_out)} playback frames")
        
        # Restore
        self.speed_multiplier = orig_speed
        self.reset(orig_idx)

# ===================================================================
# Self-test
# ===================================================================

if __name__ == "__main__":
    from features.pipeline import get_demo_user_df
    from models.risk_model import CardiacRiskModel, StressModel, SleepQualityModel
    
    logger.info("Setting up RealtimePlaybackEngine test...")
    df = get_demo_user_df()
    
    # We need models trained. Let's make mock empty models that just return defaults
    # (or we could use train_all, but we want this test to be self-contained and fast)
    cardiac = CardiacRiskModel()
    stress = StressModel()
    sleep = SleepQualityModel()
    
    models = {"cardiac": cardiac, "stress": stress, "sleep": sleep}
    
    # Run test stream for 5 iterations fast
    engine = RealtimePlaybackEngine(df, models, speed_multiplier=0.0)
    
    logger.info("Starting short stream test (5 ticks):")
    count = 0
    for state in engine.stream():
        logger.info(
            f"  {state['timestamp'].strftime('%H:%M')} | "
            f"HR: {state['raw_signals'].get('heart_rate', 0):.0f} | "
            f"Cardiac Risk: {state['risk_scores']['cardiac']['risk_score']:.2f}"
        )
        count += 1
        if count >= 5:
            engine.stop()
            break
            
    logger.success("Realtime engine test passed ✓")
