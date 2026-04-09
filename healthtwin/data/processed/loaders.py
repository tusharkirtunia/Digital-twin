"""
HealthTwin — Data Loaders Module
=================================
Standardized data loading and validation for all datasets used in the
HealthTwin digital twin pipeline.

Each loader returns a clean pd.DataFrame with:
  - `timestamp` column (datetime64)
  - `user_id` column (string)
  - Signal/feature columns named consistently

Supported datasets:
  1. UCI Heart Disease (Cleveland) — cardiac risk factors
  2. PAMAP2 — physical activity monitoring (IMU + HR)
  3. WESAD — wearable stress/affect detection (ECG, EDA, TEMP)
  4. Sleep-EDF (PhysioNet) — sleep stage classification
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import decimate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UCI_COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

PAMAP2_ACTIVITY_MAP = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

WESAD_LABELS = {
    0: "undefined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "undefined_5",
    6: "undefined_6",
    7: "undefined_7",
}

SLEEP_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # N3 = stages 3+4 combined
    "Sleep stage R": 4,
    "Movement time": 0,
    "Sleep stage ?": 0,
}


# ===================================================================
# 1. UCI Heart Disease (Cleveland)
# ===================================================================

def load_uci_heart(path: str) -> pd.DataFrame:
    """
    Load and clean the UCI Heart Disease (Cleveland) dataset.

    Parameters
    ----------
    path : str
        Path to the ``heart_cleveland.csv`` (or ``processed.cleveland.data``)
        file.  The file uses no header row and ``?`` for missing values.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns matching ``UCI_COLUMN_NAMES``,
        synthetic daily timestamps starting 2023-01-01, and a ``user_id``
        column.
    """
    logger.info(f"Loading UCI Heart Disease from {path}")
    df = pd.read_csv(
        path,
        header=None,
        names=UCI_COLUMN_NAMES,
        na_values="?",
    )

    # --- Impute missing values with column medians ----------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"  {col}: imputed {n_missing} NaN → median {median_val:.2f}")

    # --- Binarise target (0 = no disease, 1 = disease present) ----------
    df["target"] = (df["target"] > 0).astype(int)

    # --- Synthetic timestamps & user_id ---------------------------------
    start = pd.Timestamp("2023-01-01")
    df["timestamp"] = pd.date_range(start, periods=len(df), freq="D")
    df["user_id"] = "uci_patient"

    logger.success(
        f"UCI Heart Disease loaded: {df.shape[0]} rows, "
        f"target distribution: {df['target'].value_counts().to_dict()}"
    )
    return df


# ===================================================================
# 2. PAMAP2 Physical Activity
# ===================================================================

def load_pamap2(
    path: str,
    subject_ids: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Load PAMAP2 physical-activity dataset from raw ``.dat`` protocol files.

    Parameters
    ----------
    path : str
        Directory containing ``Protocol/subjectN.dat`` files.
    subject_ids : list[int], optional
        Which subject numbers to load (e.g. ``[1, 2, 5]``).
        If ``None``, loads all found files.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, user_id, activity_id, activity_name,
        heart_rate, hand_acc_x/y/z, chest_acc_x/y/z.
        Resampled to 1 Hz (mean aggregation).
    """
    logger.info(f"Loading PAMAP2 from {path}")
    protocol_dir = Path(path) / "Protocol"

    if not protocol_dir.exists():
        # Try the path directly if Protocol subdir is absent
        protocol_dir = Path(path)

    all_frames: list[pd.DataFrame] = []

    # PAMAP2 columns (54 total per row)
    # Col 0: timestamp (s), Col 1: activityID, Col 2: heart_rate
    # Then 3 IMU blocks of 17 columns each: hand (3-19), chest (20-36), ankle (37-53)
    # Within each IMU block: temp(0), acc16_x/y/z(1-3), acc6_x/y/z(4-6),
    #   gyro_x/y/z(7-9), magn_x/y/z(10-12), orientation(13-16)

    for dat_file in sorted(protocol_dir.glob("subject*.dat")):
        subj_num = int(dat_file.stem.replace("subject", ""))
        if subject_ids is not None and subj_num not in subject_ids:
            continue

        logger.debug(f"  Reading {dat_file.name}")
        raw = pd.read_csv(dat_file, sep=r"\s+", header=None)

        subj_df = pd.DataFrame({
            "timestamp_s": raw.iloc[:, 0],
            "activity_id": raw.iloc[:, 1],
            "heart_rate": raw.iloc[:, 2],
            # Hand accelerometer (16g) — columns 4,5,6 (0-indexed)
            "hand_acc_x": raw.iloc[:, 4],
            "hand_acc_y": raw.iloc[:, 5],
            "hand_acc_z": raw.iloc[:, 6],
            # Chest accelerometer (16g) — columns 21,22,23
            "chest_acc_x": raw.iloc[:, 21],
            "chest_acc_y": raw.iloc[:, 22],
            "chest_acc_z": raw.iloc[:, 23],
        })

        subj_df["user_id"] = f"pamap2_s{subj_num:02d}"

        # Map activity IDs to names
        subj_df["activity_name"] = (
            subj_df["activity_id"]
            .map(PAMAP2_ACTIVITY_MAP)
            .fillna("unknown")
        )

        # Replace NaN marker (heart_rate uses NaN natively)
        subj_df.replace({"heart_rate": {0: np.nan}}, inplace=True)

        # Convert timestamp_s (seconds since start) to datetime
        t0 = pd.Timestamp("2023-01-15 08:00:00")
        subj_df["timestamp"] = t0 + pd.to_timedelta(subj_df["timestamp_s"], unit="s")
        subj_df.drop(columns=["timestamp_s"], inplace=True)

        # Resample to 1 Hz using mean aggregation
        subj_df = subj_df.set_index("timestamp")
        numeric_cols_local = subj_df.select_dtypes(include=[np.number]).columns
        resampled = subj_df[numeric_cols_local].resample("1s").mean()
        # Forward-fill activity labels
        resampled["activity_id"] = subj_df["activity_id"].resample("1s").ffill()
        resampled["activity_name"] = (
            resampled["activity_id"]
            .map(PAMAP2_ACTIVITY_MAP)
            .fillna("unknown")
        )
        resampled["user_id"] = f"pamap2_s{subj_num:02d}"
        resampled = resampled.reset_index()

        all_frames.append(resampled)
        logger.debug(f"    subject {subj_num}: {len(resampled)} rows (1 Hz)")

    if not all_frames:
        raise FileNotFoundError(
            f"No PAMAP2 subject files found in {protocol_dir}"
        )

    result = pd.concat(all_frames, ignore_index=True)
    logger.success(
        f"PAMAP2 loaded: {result.shape[0]} rows, "
        f"{result['user_id'].nunique()} subjects"
    )
    return result


# ===================================================================
# 3. WESAD — Wearable Stress & Affect Detection
# ===================================================================

def load_wesad(path: str, subject_id: str) -> pd.DataFrame:
    """
    Load a single WESAD subject's chest-worn sensor data from pickle.

    Parameters
    ----------
    path : str
        Base WESAD directory containing ``S2/``, ``S3/``, … subdirs.
    subject_id : str
        Subject folder name, e.g. ``"S2"``.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, user_id, ecg, eda, emg, resp, temp, label.
        Downsampled from 700 Hz to 10 Hz via ``scipy.signal.decimate``.
    """
    pkl_path = Path(path) / subject_id / f"{subject_id}.pkl"
    logger.info(f"Loading WESAD subject {subject_id} from {pkl_path}")

    if not pkl_path.exists():
        raise FileNotFoundError(f"WESAD pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Chest signals are sampled at 700 Hz
    chest = data["signal"]["chest"]
    label = data["label"]  # shape (N, 1) at 700 Hz

    # Extract individual signals — each is (N, 1) or (N,)
    ecg_raw = np.squeeze(chest["ECG"])
    eda_raw = np.squeeze(chest["EDA"])
    emg_raw = np.squeeze(chest["EMG"])
    resp_raw = np.squeeze(chest["Resp"])
    temp_raw = np.squeeze(chest["Temp"])
    label_raw = np.squeeze(label)

    # Downsample 700 Hz → 10 Hz  (factor = 70)
    # decimate requires integer factor; 700/10 = 70
    downsample_factor = 70

    ecg_ds = decimate(ecg_raw, downsample_factor, zero_phase=True)
    eda_ds = decimate(eda_raw, downsample_factor, zero_phase=True)
    emg_ds = decimate(emg_raw, downsample_factor, zero_phase=True)
    resp_ds = decimate(resp_raw, downsample_factor, zero_phase=True)
    temp_ds = decimate(temp_raw, downsample_factor, zero_phase=True)
    # For labels, just subsample (no filtering needed)
    label_ds = label_raw[::downsample_factor]

    # Align lengths (decimate can produce ±1 sample difference)
    min_len = min(len(ecg_ds), len(eda_ds), len(emg_ds),
                  len(resp_ds), len(temp_ds), len(label_ds))
    ecg_ds = ecg_ds[:min_len]
    eda_ds = eda_ds[:min_len]
    emg_ds = emg_ds[:min_len]
    resp_ds = resp_ds[:min_len]
    temp_ds = temp_ds[:min_len]
    label_ds = label_ds[:min_len]

    # Build timestamps at 10 Hz
    start = pd.Timestamp("2023-02-01 09:00:00")
    timestamps = pd.date_range(start, periods=min_len, freq="100ms")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "user_id": f"wesad_{subject_id}",
        "ecg": ecg_ds,
        "eda": eda_ds,
        "emg": emg_ds,
        "resp": resp_ds,
        "temp": temp_ds,
        "label": label_ds.astype(int),
    })

    logger.success(
        f"WESAD {subject_id} loaded: {df.shape[0]} rows @ 10 Hz, "
        f"label distribution: {df['label'].value_counts().to_dict()}"
    )
    return df


# ===================================================================
# 4. Sleep-EDF (PhysioNet)
# ===================================================================

def load_sleep_edf(path: str) -> pd.DataFrame:
    """
    Load Sleep-EDF cassette data: PSG EEG signals + hypnogram labels.

    Parameters
    ----------
    path : str
        Directory containing ``*-PSG.edf`` and ``*-Hypnogram.edf`` files.

    Returns
    -------
    pd.DataFrame
        One row per 30-second epoch with columns: timestamp, user_id,
        eeg_fpz_cz_energy, eeg_pz_oz_energy, sleep_stage.
    """
    import pyedflib  # lazy import — not always available

    edf_dir = Path(path)
    psg_files = sorted(edf_dir.rglob("*PSG.edf"))

    if not psg_files:
        raise FileNotFoundError(f"No PSG.edf files found in {edf_dir}")

    all_frames: list[pd.DataFrame] = []

    for psg_file in psg_files:
        # Derive subject ID and locate matching hypnogram
        stem = psg_file.stem  # e.g. SC4001E0-PSG
        subj_id = stem.split("-")[0]  # SC4001E0
        hyp_file = psg_file.parent / f"{stem.replace('PSG', 'Hypnogram')}.edf"

        if not hyp_file.exists():
            # Try alternative naming patterns
            hyp_candidates = list(psg_file.parent.glob(f"{subj_id}*Hypnogram*"))
            if hyp_candidates:
                hyp_file = hyp_candidates[0]
            else:
                logger.warning(f"No hypnogram found for {psg_file.name}, skipping")
                continue

        logger.debug(f"  Processing {psg_file.name}")

        # --- Read PSG signals -------------------------------------------
        try:
            psg = pyedflib.EdfReader(str(psg_file))
        except Exception as e:
            logger.error(f"Failed to read {psg_file.name}: {e}")
            continue

        labels = [psg.getLabel(i) for i in range(psg.signals_in_file)]
        fs = psg.getSampleFrequency(0)

        # Find EEG channels by name pattern
        fpz_idx = next(
            (i for i, l in enumerate(labels) if "Fpz" in l or "FPZ" in l),
            None,
        )
        pz_idx = next(
            (i for i, l in enumerate(labels) if "Pz" in l or "PZ" in l),
            None,
        )

        if fpz_idx is None or pz_idx is None:
            logger.warning(
                f"EEG channels not found in {psg_file.name} "
                f"(available: {labels}), skipping"
            )
            psg.close()
            continue

        fpz_signal = psg.readSignal(fpz_idx)
        pz_signal = psg.readSignal(pz_idx)
        psg.close()

        # --- Read hypnogram ---------------------------------------------
        try:
            hyp = pyedflib.EdfReader(str(hyp_file))
        except Exception as e:
            logger.error(f"Failed to read hypnogram for {subj_id}: {e}")
            continue

        annotations = hyp.readAnnotations()
        hyp.close()

        # Build epoch-level labels (30-second epochs)
        epoch_duration = 30  # seconds
        samples_per_epoch = int(fs * epoch_duration)
        n_epochs = len(fpz_signal) // samples_per_epoch

        # Parse annotation onsets and stages
        stage_labels = np.zeros(n_epochs, dtype=int)
        for onset, _, annotation in zip(
            annotations[0], annotations[1], annotations[2]
        ):
            epoch_idx = int(float(onset) / epoch_duration)
            stage = SLEEP_STAGE_MAP.get(annotation, 0)
            duration_epochs = max(1, int(float(annotations[1][0]) / epoch_duration))
            end_idx = min(epoch_idx + duration_epochs, n_epochs)
            stage_labels[epoch_idx:end_idx] = stage

        # --- Compute epoch features: mean absolute signal energy --------
        fpz_energy = np.array([
            np.mean(np.abs(
                fpz_signal[i * samples_per_epoch:(i + 1) * samples_per_epoch]
            ))
            for i in range(n_epochs)
        ])
        pz_energy = np.array([
            np.mean(np.abs(
                pz_signal[i * samples_per_epoch:(i + 1) * samples_per_epoch]
            ))
            for i in range(n_epochs)
        ])

        # Build DataFrame
        start_time = pd.Timestamp("2023-03-01 21:00:00")
        timestamps = pd.date_range(
            start_time, periods=n_epochs, freq=f"{epoch_duration}s"
        )

        epoch_df = pd.DataFrame({
            "timestamp": timestamps[:n_epochs],
            "user_id": f"sleep_{subj_id}",
            "eeg_fpz_cz_energy": fpz_energy,
            "eeg_pz_oz_energy": pz_energy,
            "sleep_stage": stage_labels,
        })

        all_frames.append(epoch_df)
        logger.debug(
            f"    {subj_id}: {n_epochs} epochs, "
            f"stages: {np.unique(stage_labels, return_counts=True)}"
        )

    if not all_frames:
        raise FileNotFoundError(f"No valid Sleep-EDF files processed in {edf_dir}")

    result = pd.concat(all_frames, ignore_index=True)
    logger.success(
        f"Sleep-EDF loaded: {result.shape[0]} epochs, "
        f"{result['user_id'].nunique()} subjects"
    )
    return result


# ===================================================================
# 5. DataFrame Validator
# ===================================================================

def validate_dataframe(df: pd.DataFrame, name: str) -> bool:
    """
    Validate a loaded DataFrame for pipeline compatibility.

    Checks
    ------
    - Required columns ``timestamp`` and ``user_id`` exist.
    - DataFrame is non-empty.
    - Logs shape, date range, and per-column NaN percentage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    name : str
        Human-readable dataset name for logging.

    Returns
    -------
    bool
        ``True`` if valid.

    Raises
    ------
    ValueError
        If required columns are missing or DataFrame is empty.
    """
    logger.info(f"Validating DataFrame: {name}")

    if df.empty:
        raise ValueError(f"[{name}] DataFrame is empty!")

    required = {"timestamp", "user_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{name}] Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Log summary statistics
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Users: {df['user_id'].nunique()}")

    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        logger.info(
            f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}"
        )
    else:
        logger.warning(
            f"  timestamp column is not datetime64 "
            f"(dtype={df['timestamp'].dtype})"
        )

    # NaN report
    nan_pct = (df.isna().sum() / len(df) * 100).round(2)
    cols_with_nan = nan_pct[nan_pct > 0].sort_values(ascending=False)
    if not cols_with_nan.empty:
        logger.info("  NaN percentages:")
        for col, pct in cols_with_nan.items():
            logger.info(f"    {col}: {pct}%")
    else:
        logger.info("  No NaN values found ✓")

    logger.success(f"  [{name}] validation PASSED ✓")
    return True


# ===================================================================
# Quick-test entrypoint
# ===================================================================

if __name__ == "__main__":
    """Quick integration test — tries to load whichever datasets exist."""
    import sys

    raw_base = Path(__file__).resolve().parent.parent / "raw"
    logger.info(f"Looking for datasets in {raw_base}")

    # UCI Heart Disease
    uci_path = raw_base / "uci_heart" / "heart_cleveland.csv"
    if uci_path.exists():
        df = load_uci_heart(str(uci_path))
        validate_dataframe(df, "UCI Heart Disease")
    else:
        logger.warning(f"UCI Heart Disease not found at {uci_path}")

    # PAMAP2
    pamap2_path = raw_base / "pamap2" / "PAMAP2_Dataset"
    if pamap2_path.exists():
        df = load_pamap2(str(pamap2_path), subject_ids=[1])
        validate_dataframe(df, "PAMAP2")
    else:
        logger.warning(f"PAMAP2 not found at {pamap2_path}")

    # WESAD
    wesad_path = raw_base / "wesad"
    if (wesad_path / "S2").exists():
        df = load_wesad(str(wesad_path), "S2")
        validate_dataframe(df, "WESAD")
    else:
        logger.warning(f"WESAD not found at {wesad_path}")

    # Sleep-EDF
    sleep_path = raw_base / "sleep_edf"
    if sleep_path.exists() and any(sleep_path.rglob("*PSG.edf")):
        df = load_sleep_edf(str(sleep_path))
        validate_dataframe(df, "Sleep-EDF")
    else:
        logger.warning(f"Sleep-EDF EDF files not found in {sleep_path}")

    logger.info("Loader test complete.")
