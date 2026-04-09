"""
HealthTwin — HRV Feature Extractor
=====================================
Processes raw ECG signals from WESAD (or any source) into Heart Rate
Variability (HRV) features using neurokit2.

HRV is a key biomarker for autonomic nervous system activity:
- Low HRV → sympathetic dominance → stress / fatigue
- High HRV → parasympathetic dominance → recovery / relaxation

Usage::

    ext = HRVFeatureExtractor()
    hrv_dict = ext.extract_from_ecg(ecg_signal, sampling_rate=700)
    stress = ext.compute_stress_index(hrv_dict)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Keys we always return (NaN-safe)
HRV_KEYS = [
    # Time domain
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_pNN50",
    "HRV_pNN20",
    # Frequency domain
    "HRV_LF",
    "HRV_HF",
    "HRV_LFHF",
    # Nonlinear
    "HRV_SD1",
    "HRV_SD2",
    "HRV_ApEn",
]


def _nan_hrv_dict() -> dict[str, float]:
    """Return an HRV dict filled with NaN for all standard keys."""
    return {k: np.nan for k in HRV_KEYS}


class HRVFeatureExtractor:
    """
    Extract HRV features from raw ECG signals via neurokit2.

    All methods are designed to be fault-tolerant — noisy or short
    segments return NaN dicts rather than raising exceptions.
    """

    def __init__(self) -> None:
        # Lazy-import neurokit2 to avoid heavy startup cost
        self._nk: Optional[object] = None

    @property
    def nk(self):
        """Lazy-load neurokit2."""
        if self._nk is None:
            import neurokit2 as nk
            self._nk = nk
        return self._nk

    # ------------------------------------------------------------------
    # METHOD 1 — Single-segment HRV from ECG
    # ------------------------------------------------------------------

    def extract_from_ecg(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: int = 700,
    ) -> dict[str, float]:
        """
        Extract HRV features from a single ECG segment.

        Parameters
        ----------
        ecg_signal : np.ndarray
            Raw ECG signal (1-D).
        sampling_rate : int
            Sampling rate in Hz (WESAD chest = 700 Hz).

        Returns
        -------
        dict[str, float]
            HRV features keyed by standard neurokit2 names.
            Returns NaN values if the signal is too short or noisy.
        """
        nk = self.nk
        min_length = sampling_rate * 10  # at least 10 seconds

        if len(ecg_signal) < min_length:
            logger.debug(
                f"ECG segment too short ({len(ecg_signal)} < {min_length}), "
                "returning NaN"
            )
            return _nan_hrv_dict()

        try:
            # Clean the ECG signal
            ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)

            # Detect R-peaks
            _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
            peak_indices = rpeaks.get("ECG_R_Peaks", [])

            if len(peak_indices) < 5:
                logger.debug(
                    f"Too few R-peaks detected ({len(peak_indices)}), "
                    "returning NaN"
                )
                return _nan_hrv_dict()

            # Compute full HRV
            hrv_result = nk.hrv(
                peak_indices,
                sampling_rate=sampling_rate,
                show=False,
            )

            # Extract our target metrics from the DataFrame row
            out = {}
            for key in HRV_KEYS:
                if key in hrv_result.columns:
                    val = hrv_result[key].iloc[0]
                    out[key] = float(val) if not pd.isna(val) else np.nan
                else:
                    out[key] = np.nan

            return out

        except Exception as e:
            logger.debug(f"HRV extraction failed: {e}")
            return _nan_hrv_dict()

    # ------------------------------------------------------------------
    # METHOD 2 — Windowed HRV (sliding 5-min windows)
    # ------------------------------------------------------------------

    def extract_windowed(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: int = 700,
        window_seconds: int = 300,
        overlap: float = 0.5,
    ) -> pd.DataFrame:
        """
        Compute HRV features in sliding windows over a long ECG recording.

        Parameters
        ----------
        ecg_signal : np.ndarray
            Full ECG recording.
        sampling_rate : int
            Sampling frequency in Hz.
        window_seconds : int
            Window length in seconds (default: 300 = 5 min).
        overlap : float
            Fraction of overlap between windows (default: 0.5 = 50%).

        Returns
        -------
        pd.DataFrame
            One row per window, columns = HRV metric names +
            ``window_start_seconds``.
        """
        window_samples = window_seconds * sampling_rate
        step_samples = int(window_samples * (1 - overlap))
        total_samples = len(ecg_signal)

        rows: list[dict] = []
        start = 0

        while start + window_samples <= total_samples:
            segment = ecg_signal[start : start + window_samples]
            hrv = self.extract_from_ecg(segment, sampling_rate=sampling_rate)
            hrv["window_start_seconds"] = start / sampling_rate
            rows.append(hrv)
            start += step_samples

        if not rows:
            logger.warning(
                f"ECG too short for any {window_seconds}s window "
                f"({total_samples / sampling_rate:.0f}s available)"
            )
            empty = _nan_hrv_dict()
            empty["window_start_seconds"] = 0.0
            return pd.DataFrame([empty])

        logger.debug(
            f"Extracted HRV from {len(rows)} × {window_seconds}s windows"
        )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # METHOD 3 — Stress Index from HRV
    # ------------------------------------------------------------------

    def compute_stress_index(self, hrv_dict: dict[str, float]) -> float:
        """
        Compute a normalised stress index (0–1) from LF/HF ratio.

        Interpretation:
        - LF/HF > 2.0 → high sympathetic activation → high stress
        - LF/HF < 0.5 → parasympathetic dominance → relaxed

        Uses a sigmoid to map LF/HF (typically 0–6) to 0–1.

        Parameters
        ----------
        hrv_dict : dict
            Must contain ``HRV_LFHF`` key.

        Returns
        -------
        float
            Stress index in [0, 1]. Returns 0.5 if LF/HF unavailable.
        """
        lfhf = hrv_dict.get("HRV_LFHF", np.nan)
        if np.isnan(lfhf):
            return 0.5  # neutral default

        # Sigmoid: centre at 1.5, steepness 2
        stress = 1.0 / (1.0 + np.exp(-2.0 * (lfhf - 1.5)))
        return float(np.clip(stress, 0.0, 1.0))

    # ------------------------------------------------------------------
    # METHOD 4 — Batch process an entire WESAD DataFrame
    # ------------------------------------------------------------------

    def batch_process_wesad(
        self,
        wesad_df: pd.DataFrame,
        window_seconds: int = 300,
        ecg_sampling_rate: int = 10,
    ) -> pd.DataFrame:
        """
        Process all subjects in a WESAD DataFrame, computing windowed HRV.

        Parameters
        ----------
        wesad_df : pd.DataFrame
            Output of ``load_wesad()`` with columns: timestamp, user_id,
            ecg, eda, emg, resp, temp, label.
        window_seconds : int
            HRV analysis window in seconds.
        ecg_sampling_rate : int
            ECG sampling rate after downsampling (default: 10 Hz for
            WESAD post-downsample).

        Returns
        -------
        pd.DataFrame
            Original WESAD data (downsampled to match window timestamps)
            merged with HRV features and stress_index.
        """
        logger.info(
            f"Batch processing WESAD: "
            f"{wesad_df['user_id'].nunique()} subjects"
        )

        all_results: list[pd.DataFrame] = []

        for uid, udf in wesad_df.groupby("user_id"):
            udf = udf.sort_values("timestamp").reset_index(drop=True)
            ecg_signal = udf["ecg"].values

            # Extract windowed HRV
            hrv_df = self.extract_windowed(
                ecg_signal,
                sampling_rate=ecg_sampling_rate,
                window_seconds=window_seconds,
            )

            # Map window start times back to timestamps
            hrv_df["timestamp"] = udf["timestamp"].iloc[0] + pd.to_timedelta(
                hrv_df["window_start_seconds"], unit="s"
            )
            hrv_df["user_id"] = uid

            # Add stress index
            hrv_df["stress_index"] = hrv_df.apply(
                lambda row: self.compute_stress_index(row.to_dict()),
                axis=1,
            )

            # Merge back with original data using nearest timestamp
            hrv_df = hrv_df.drop(columns=["window_start_seconds"])
            merged = pd.merge_asof(
                udf.sort_values("timestamp"),
                hrv_df.sort_values("timestamp"),
                on="timestamp",
                by="user_id",
                direction="nearest",
            )

            all_results.append(merged)
            logger.debug(f"  [{uid}] → {len(hrv_df)} HRV windows")

        result = pd.concat(all_results, ignore_index=True)
        logger.success(
            f"WESAD batch processing complete: "
            f"{result.shape[0]} rows, "
            f"{len([c for c in result.columns if c.startswith('HRV_')])} HRV features"
        )
        return result


# ======================================================================
# Self-test with synthetic ECG-like signal
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)
    ext = HRVFeatureExtractor()

    # Generate a simple synthetic "ECG-like" signal (not realistic but
    # enough to validate the pipeline doesn't crash)
    fs = 100  # lower rate for fast testing
    duration = 600  # 10 minutes
    t = np.arange(0, duration, 1 / fs)

    # Simulated heartbeat at ~72 bpm (1.2 Hz)
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(len(t))
    # Add QRS-like spikes
    for beat_time in np.arange(0, duration, 0.833):  # ~72 bpm
        idx = int(beat_time * fs)
        if idx + 5 < len(ecg):
            ecg[idx : idx + 5] += 3.0

    logger.info(f"Synthetic ECG: {len(ecg)} samples @ {fs} Hz ({duration}s)")

    # Test single-segment extraction
    hrv_single = ext.extract_from_ecg(ecg[:fs * 120], sampling_rate=fs)
    logger.info(f"Single-segment HRV: {hrv_single}")

    # Test windowed extraction
    hrv_windowed = ext.extract_windowed(ecg, sampling_rate=fs, window_seconds=120)
    logger.info(f"Windowed HRV: {hrv_windowed.shape}")
    logger.info(f"Columns: {list(hrv_windowed.columns)}")

    # Test stress index
    stress = ext.compute_stress_index(hrv_single)
    logger.info(f"Stress index: {stress:.3f}")
