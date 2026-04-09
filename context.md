# 🧬 AI Digital Twin — Copilot/Codex Prompt Bible
### macOS · pyenv + venv · GitHub Copilot in VS Code · 48-hour Hackathon

---

> **How to use this file**
> - **Copilot Chat prompts** → open Copilot Chat sidebar (`Cmd+Shift+I`), paste the prompt under `[CHAT]`
> - **Inline prompts** → paste as a comment directly in the file, press `Tab` to accept autocomplete
> - **Terminal commands** → run exactly as written in your project root
> - Paste `CONTEXT.md` content into every new Copilot Chat session before any prompt

---

## ⚡ BEFORE EVERYTHING — Create Your CONTEXT.md

Create a file called `CONTEXT.md` in your project root.
**Paste this at the start of EVERY new Copilot Chat session.**

```
You are working on a hackathon project called "HealthTwin" — an AI-powered Digital Twin for Personal Health.

PROJECT STACK:
- Language: Python 3.11
- Environment: pyenv + venv, macOS
- UI: Streamlit
- ML: XGBoost, scikit-learn, optional PyTorch LSTM
- XAI: SHAP, DiCE-ml
- Feature engineering: tsfresh, neurokit2, pandas rolling windows
- Physiology: neurokit2 for ECG/HRV processing
- Visualization: Plotly, matplotlib

FOLDER STRUCTURE:
healthtwin/
├── data/raw/          # downloaded datasets
├── data/processed/    # cleaned, merged, normalized
├── features/          # rolling.py, baseline.py, hrv.py
├── models/            # risk_model.py, sleep_model.py, counterfactual.py
├── explain/           # shap_explainer.py
├── app/               # streamlit_app.py
├── notebooks/         # eda.ipynb
├── tests/             # unit tests
└── CONTEXT.md

DATASETS IN USE:
1. UCI Heart Disease (Cleveland) — cardiac risk factors
2. WESAD — wearable stress/affect detection (ECG, EDA, TEMP, ACC)
3. PAMAP2 — physical activity monitoring with IMU + HR
4. Sleep-EDF (PhysioNet) — sleep stage classification
5. PMData — longitudinal wearable + wellness surveys

CORE CONCEPTS:
- Rolling time-series features: 24h, 7d, 30d windows with statistical aggregates
- Personal baseline normalization: z-score each metric against user's own rolling mean/std
- Simulated real-time playback: chronological row-by-row replay with time.sleep() in Streamlit
- What-if simulator: DiCE counterfactual engine on trained XGBoost model
- XAI: SHAP waterfall charts + LLM-generated plain-English health brief

CURRENT PHASE: [UPDATE THIS as you progress]
LAST FILE MODIFIED: [UPDATE THIS]
DO NOT rewrite files I haven't asked you to change.
Always write production-quality, commented Python code.
```

---

---

# PHASE 1 — Environment & Data Pipeline
## ⏱ Hours 0–4

---

### STEP 1.1 — Project Scaffold & Environment Setup

**Run in terminal:**
```bash
# Install Python 3.11 via pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Create and activate venv
python -m venv .venv
source .venv/bin/activate

# Confirm
python --version  # should show 3.11.9

# Create project structure
mkdir -p healthtwin/{data/{raw,processed},features,models,explain,app,notebooks,tests}
cd healthtwin
touch features/{__init__,rolling,baseline,hrv}.py
touch models/{__init__,risk_model,sleep_model,counterfactual}.py
touch explain/{__init__,shap_explainer}.py
touch app/streamlit_app.py
touch requirements.txt
code .  # open in VS Code
```

---

### STEP 1.2 — Requirements File

**[INLINE PROMPT]** — Open `requirements.txt`, paste this comment then let Copilot fill or just paste directly:

```
# Core data
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0

# Feature engineering
tsfresh==0.20.2
tsfel==0.1.9

# Physiological signal processing
neurokit2==0.2.7
pyEDFlib==0.1.36

# Machine learning
scikit-learn==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
imbalanced-learn==0.12.2

# Deep learning (optional LSTM branch)
torch==2.3.0
tsai==0.3.9

# Explainability
shap==0.45.1
dice-ml==0.11

# Visualization
plotly==5.22.0
matplotlib==3.9.0
seaborn==0.13.2

# Dashboard
streamlit==1.35.0
streamlit-extras==0.4.3

# Notebook
jupyterlab==4.2.1

# Utilities
python-dotenv==1.0.1
tqdm==4.66.4
PyYAML==6.0.1
loguru==0.7.2
```

**Run:**
```bash
pip install -r requirements.txt
```

---

### STEP 1.3 — Dataset Downloads

**Run each block in terminal from your `healthtwin/data/raw/` directory:**

```bash
cd data/raw

# ── 1. UCI Heart Disease (Cleveland) ──
mkdir uci_heart && cd uci_heart
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
# Rename for clarity
mv processed.cleveland.data heart_cleveland.csv
cd ..

# ── 2. PAMAP2 Physical Activity ──
mkdir pamap2 && cd pamap2
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip
unzip PAMAP2_Dataset.zip
rm PAMAP2_Dataset.zip
cd ..

# ── 3. Sleep-EDF (PhysioNet — subset, no login needed) ──
mkdir sleep_edf && cd sleep_edf
# Cassette portion, 2 subjects to start (expand later)
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/ \
  --accept "SC4001*,SC4002*" -P .
cd ..

# ── 4. WESAD ──
# Requires manual download (free registration at uni-siegen.de)
mkdir wesad
echo "ACTION REQUIRED: Download WESAD from:"
echo "https://uni-siegen.wpengine.com/lab/research/wearable-sensor-data/wesad/"
echo "Extract into data/raw/wesad/"

# ── 5. PMData ──
mkdir pmdata
echo "ACTION REQUIRED: Download PMData from:"
echo "https://datasets.simula.no/pmdata/"
echo "Extract into data/raw/pmdata/"

cd ../..  # back to healthtwin root
```

> **Note on WESAD & PMData:** Both are free but require a quick form/registration.
> Do these manually in parallel while your pip install runs.

---

### STEP 1.4 — Data Loader Module

**[CHAT PROMPT — Copilot Chat `Cmd+Shift+I`]**

```
[PASTE YOUR CONTEXT.MD FIRST]

Create a file `data/processed/loaders.py`.

Write a Python module with the following functions. Each must return a clean pandas DataFrame with a `timestamp` column (datetime), a `user_id` column (string), and signal columns named consistently.

Functions to write:

1. `load_uci_heart(path: str) -> pd.DataFrame`
   - Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
   - Column names from UCI spec, target=1 is disease presence
   - Handle missing values marked as "?" — replace with np.nan, then impute median
   - Add synthetic timestamps starting 2023-01-01, one row per day per user_id

2. `load_pamap2(path: str, subject_ids: list = None) -> pd.DataFrame`
   - Raw columns: timestamp, activity_id, heart_rate, then IMU columns for hand/chest/ankle
   - Keep: timestamp, subject_id (from filename), activity_id, heart_rate, hand_acc_x/y/z, chest_acc_x/y/z
   - Resample to 1Hz using mean aggregation
   - Map activity_id to activity_name using the PAMAP2 protocol (lying=1, sitting=2, standing=3, walking=4, running=5, cycling=6, etc.)

3. `load_wesad(path: str, subject_id: str) -> pd.DataFrame`
   - WESAD is a .pkl file per subject — load with pickle
   - Extract chest-worn signals: ECG, EDA, EMG, RESP, TEMP (all sampled at 700Hz)
   - Downsample to 10Hz using scipy.signal.decimate
   - Add label column (1=baseline, 2=stress, 3=amusement, 4=meditation)
   - Return with columns: timestamp, subject_id, ecg, eda, emg, resp, temp, label

4. `load_sleep_edf(path: str) -> pd.DataFrame`
   - Read PSG .edf file using pyEDFlib
   - Extract EEG Fpz-Cz channel and EEG Pz-Oz channel sampled at 100Hz
   - Load hypnogram from -Hypnogram.edf file for sleep stage labels
   - Resample to 1-minute epochs with mean absolute signal energy as feature
   - Sleep stages: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM

5. `validate_dataframe(df: pd.DataFrame, name: str) -> bool`
   - Check required columns exist (timestamp, user_id)
   - Log row count, date range, NaN percentage per column
   - Return True if valid, raise ValueError with descriptive message if not

Use loguru for all logging. Include type hints on all functions. Add docstrings.
```

---

### STEP 1.5 — Verify Data Loading

**[INLINE PROMPT]** — Open `notebooks/eda.ipynb`, new cell:

```python
# Load and validate all datasets, print shape, date range, and null% for each
# Use loaders from data/processed/loaders.py
# Plot one sample time series from each dataset using plotly
```

---

---

# PHASE 2 — Feature Engineering (The Twin's Brain)
## ⏱ Hours 4–10

---

### STEP 2.1 — Rolling Time-Series Feature Engine

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `features/rolling.py`.

This module is the core of the "digital twin" concept — it transforms raw time-series data into a rich set of rolling statistical features that capture trends, volatility, and momentum across multiple time horizons.

Write a class `RollingFeatureEngine` with:

CLASS ATTRIBUTES:
- windows = [96, 672, 2880]  # 24h, 7d, 30d in 15-min intervals
- target_columns = ['heart_rate', 'hrv_rmssd', 'eda_mean', 'resp_rate', 'temp', 'activity_count', 'sleep_hours']

METHOD 1: `fit_transform(self, df: pd.DataFrame, user_id: str) -> pd.DataFrame`
For each column in target_columns (if present) and each window in windows, compute:
  - rolling mean: `{col}_{w}w_mean`
  - rolling std: `{col}_{w}w_std`
  - rolling min/max: `{col}_{w}w_min`, `{col}_{w}w_max`
  - rolling z-score vs window: `{col}_{w}w_zscore` = (current - rolling_mean) / rolling_std
  - linear trend slope over window: `{col}_{w}w_slope` using np.polyfit inside rolling.apply
  - percent change from window mean: `{col}_{w}w_pct_change`
  - lag features: `{col}_lag_1`, `{col}_lag_3`, `{col}_lag_7` (in days)

Also compute cross-signal features:
  - `sleep_hr_interaction` = sleep_hours * heart_rate (if both present)
  - `stress_activity_ratio` = eda_mean / (activity_count + 1)
  - `recovery_score` = hrv_rmssd / (heart_rate + 1) * 100

Handle NaN propagation carefully — early rows will have NaN for long windows, that's expected. Use min_periods=10 for all rolling operations.

METHOD 2: `get_feature_names(self) -> list`
Return sorted list of all generated feature column names.

METHOD 3: `get_feature_report(self, df: pd.DataFrame) -> dict`
Return dict with: total_features, nan_percentage_per_feature (top 20 worst), feature_correlation_with_target (if 'risk_label' column present).

Use pandas vectorized operations only — no Python loops over rows.
Add a `if __name__ == '__main__'` demo block that generates synthetic data and tests the engine.
```

---

### STEP 2.2 — Personal Baseline Normalization

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `features/baseline.py`.

This module implements personal baseline normalization — the key concept that makes predictions PERSONALIZED rather than population-level. Each user's metrics are normalized against their OWN historical baseline, not population averages.

Write a class `PersonalBaselineNormalizer` with:

METHOD 1: `fit(self, df: pd.DataFrame, user_id_col='user_id') -> None`
For each user and each numeric column:
  - Compute the user's personal mean and std using the FIRST 30 days of their data as baseline period
  - Store in self.baselines dict: {user_id: {col: {'mean': float, 'std': float, 'percentiles': [5,25,50,75,95]}}}
  - Also compute: personal_resting_hr (5th percentile of heart_rate during sleep hours)
  - personal_max_hr (95th percentile during active periods)
  - Log a warning if any user has fewer than 7 days of data (insufficient baseline)

METHOD 2: `transform(self, df: pd.DataFrame) -> pd.DataFrame`
For each user and each numeric column:
  - Add column `{col}_personal_zscore` = (value - personal_mean) / personal_std
  - Add column `{col}_personal_percentile` = percentile rank within user's own distribution
  - Add column `{col}_deviation_flag` = 1 if |personal_zscore| > 2.0 (anomaly), else 0
  - Add `overall_anomaly_score` = count of deviation_flags / total monitored signals

METHOD 3: `get_user_profile(self, user_id: str) -> dict`
Return a human-readable health profile dict:
{
  'user_id': ...,
  'baseline_period_days': ...,
  'resting_hr': ...,
  'typical_hrv': ...,
  'typical_sleep_hours': ...,
  'typical_stress_level': ...,
  'anomaly_history': [list of dates with high anomaly scores]
}

METHOD 4: `save(self, path: str) -> None` / `load(cls, path: str) -> PersonalBaselineNormalizer`
Serialize/deserialize with pickle + yaml metadata file alongside.

Use only pandas, numpy, scipy. Type hints throughout. Docstrings on every method.
```

---

### STEP 2.3 — HRV Feature Extractor

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `features/hrv.py`.

Write a class `HRVFeatureExtractor` that processes raw ECG signals from WESAD into HRV features using neurokit2.

METHOD 1: `extract_from_ecg(self, ecg_signal: np.ndarray, sampling_rate: int = 700) -> dict`
Using neurokit2:
  - Clean ECG: nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
  - Find R-peaks: nk.ecg_peaks()
  - Compute HRV: nk.hrv(peaks, sampling_rate=sampling_rate, show=False)
  - Return dict with these specific keys (extract from neurokit2 output):
    Time domain: HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_pNN50, HRV_pNN20
    Frequency domain: HRV_LF, HRV_HF, HRV_LFHF (LF/HF ratio — stress marker)
    Nonlinear: HRV_SD1, HRV_SD2, HRV_ApEn
  - Handle exceptions: if signal is too short or too noisy, return dict of NaN values with same keys

METHOD 2: `extract_windowed(self, ecg_signal: np.ndarray, sampling_rate: int, window_seconds: int = 300) -> pd.DataFrame`
  - Slide a 5-minute (300s) window with 50% overlap over the full signal
  - Call extract_from_ecg on each window
  - Return DataFrame with one row per window, columns = HRV metric names + 'window_start_seconds'

METHOD 3: `compute_stress_index(self, hrv_dict: dict) -> float`
  - Stress index = LF/HF ratio normalized to 0-1 scale using sigmoid
  - If LF/HF > 2.0: high stress; < 0.5: relaxed
  - Return float between 0 and 1

METHOD 4: `batch_process_wesad(self, wesad_df: pd.DataFrame) -> pd.DataFrame`
  - Group by subject_id and process each subject's ECG in 5-minute windows
  - Merge resulting HRV features back with original timestamps
  - Add 'stress_index' column from compute_stress_index
  - Return merged DataFrame

All methods must handle NaN and short segments gracefully without crashing.
```

---

### STEP 2.4 — Master Feature Pipeline

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `features/pipeline.py`.

Write a function `build_feature_matrix(raw_dfs: dict, user_id: str = 'demo_user') -> tuple[pd.DataFrame, PersonalBaselineNormalizer]`

Parameters:
  raw_dfs: dict with optional keys: 'heart', 'pamap2', 'wesad', 'sleep'
  user_id: user identifier string

Steps:
  1. For each present dataset, call the appropriate HRV/loader processing
  2. Align all DataFrames to a common 15-minute timestamp index using pd.merge_asof
  3. Apply RollingFeatureEngine.fit_transform()
  4. Apply PersonalBaselineNormalizer.fit() then .transform()
  5. Drop columns with >60% NaN
  6. Return (feature_matrix_df, fitted_normalizer)

Also write `get_demo_user_df() -> pd.DataFrame` that:
  - Loads a single PAMAP2 subject + first WESAD subject
  - Runs the full pipeline
  - Returns a ready-to-model DataFrame
  - This is used for fast testing without loading everything

Print feature matrix shape and a sample of feature names at the end.
```

---

---

# PHASE 3 — Model Core (Risk Prediction)
## ⏱ Hours 10–18

---

### STEP 3.1 — Cardiac Risk Model

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `models/risk_model.py`.

Write a class `CardiacRiskModel` using XGBoost as the primary model.

CONTEXT: We train on UCI Heart Disease (Cleveland) for the risk classification backbone, then apply the model to rolling features from wearable data for real-time risk scoring.

METHOD 1: `train(self, df: pd.DataFrame) -> dict`
Input: UCI heart disease DataFrame from loaders.py
  - Features: all columns except 'target', 'user_id', 'timestamp'
  - Target: binary 'target' column (0=no disease, 1=disease)
  - Pipeline:
    a. SimpleImputer(strategy='median') for NaN
    b. StandardScaler
    c. SMOTE for class imbalance (from imbalanced-learn)
    d. XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='auc', use_label_encoder=False)
  - Train/test split: 80/20 stratified
  - Return metrics dict: {auc, f1, precision, recall, confusion_matrix}

METHOD 2: `score_realtime(self, feature_row: pd.Series) -> dict`
Input: a single row of rolling features from the feature pipeline
  - Return: {'risk_score': float 0-1, 'risk_level': str ('Low'/'Moderate'/'High'/'Critical'), 'top_risk_factors': list of top 5 feature names by SHAP value}
  - Thresholds: <0.3=Low, 0.3-0.55=Moderate, 0.55-0.75=High, >0.75=Critical

METHOD 3: `score_timeseries(self, feature_df: pd.DataFrame) -> pd.DataFrame`
Apply score_realtime across a full DataFrame
  - Return DataFrame with added columns: risk_score, risk_level
  - Also add 7-day rolling average of risk_score: risk_score_7d_avg
  - Flag rows where risk_score > risk_score_7d_avg * 1.3 as 'risk_spike'

METHOD 4: `save(self, path: str)` / `load(cls, path: str)`
Serialize full sklearn Pipeline with joblib.

Also write a `StressModel` class (same file) that:
  - Trains on WESAD data using HRV features + EDA features
  - Target: binary (stress=1 when label==2, else 0)
  - Same XGBoost pipeline but with different feature set
  - Same score_realtime interface

And a `SleepQualityModel` class that:
  - Trains on Sleep-EDF features: mean EEG energy by stage, total sleep time, wake-after-sleep-onset
  - Target: binary good_sleep (1 if N3+REM > 40% of total sleep)
  - Returns sleep_quality_score 0-1

All three classes must share a `BaseHealthModel` abstract class with common interface.
```

---

### STEP 3.2 — Model Training Script

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `models/train_all.py` — a standalone script that trains all three models in sequence.

Script flow:
1. Load all datasets using loaders.py
2. Build feature matrices using features/pipeline.py
3. Train CardiacRiskModel, StressModel, SleepQualityModel
4. Save all models to models/saved/ directory with timestamp in filename
5. Print a final summary table (use rich library or plain print) showing model name, AUC, F1, training samples
6. Save training metrics to models/saved/metrics.yaml

Add CLI args using argparse:
  --models [cardiac|stress|sleep|all] (default: all)
  --data-path PATH (default: data/raw)
  --output-path PATH (default: models/saved)
  --verbose flag

Run command:
python -m models.train_all --models all --verbose
```

**Run:**
```bash
python -m models.train_all --models all --verbose
```

---

### STEP 3.3 — Simulated Real-Time Playback Engine

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `models/realtime_engine.py`.

This is the simulated real-time playback system — it replays a historical dataset chronologically, feeding data point-by-point into the rolling feature pipeline and model, simulating what a live wearable would do.

Write a class `RealtimePlaybackEngine`:

METHOD 1: `__init__(self, df: pd.DataFrame, models: dict, normalizer: PersonalBaselineNormalizer, speed_multiplier: float = 1.0)`
  - df: full historical DataFrame sorted by timestamp
  - models: {'cardiac': CardiacRiskModel, 'stress': StressModel, 'sleep': SleepQualityModel}
  - normalizer: fitted PersonalBaselineNormalizer
  - speed_multiplier: 1.0 = real-time, 10.0 = 10x faster, 0 = instant (no sleep)
  - Maintain an internal rolling buffer of last 2880 rows (30 days at 15-min intervals)

METHOD 2: `stream(self) -> Generator[dict, None, None]`
A Python generator that:
  - Yields one observation dict per call
  - Each dict contains: {timestamp, raw_signals, rolling_features, risk_scores: {cardiac, stress, sleep}, anomaly_flags, personal_zscores}
  - Updates internal rolling buffer with each new row
  - Recomputes rolling features incrementally (only new window, not full recompute)
  - Applies time.sleep(interval / speed_multiplier) between yields
  - Handles end of data gracefully — stops iteration

METHOD 3: `get_current_state(self) -> dict`
Return latest state snapshot (same format as stream yield) without advancing

METHOD 4: `reset(self, start_index: int = 0)`
Reset playback to any point in the dataset

METHOD 5: `export_playback_session(self, output_path: str)`
Run full playback at speed_multiplier=0, save all outputs to CSV for debugging

The engine must be thread-safe so Streamlit can call stream() from a background thread.
Use threading.Lock() on the rolling buffer.
```

---

---

# PHASE 4 — Explainability + What-If Simulator
## ⏱ Hours 18–28

---

### STEP 4.1 — SHAP Explainer

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `explain/shap_explainer.py`.

Write a class `HealthExplainer`:

METHOD 1: `__init__(self, models: dict)`
  - Create a shap.TreeExplainer for each XGBoost model in models dict
  - Store feature names per model

METHOD 2: `explain_prediction(self, model_name: str, feature_row: pd.Series) -> dict`
Returns:
{
  'shap_values': dict of {feature_name: shap_value},
  'base_value': float,
  'prediction': float,
  'top_drivers_positive': list of top 5 (feature, value, shap) tuples pushing risk UP,
  'top_drivers_negative': list of top 5 (feature, value, shap) tuples pushing risk DOWN,
  'plotly_waterfall_fig': plotly.graph_objects.Figure  # waterfall chart
}

The plotly waterfall chart should:
  - Show top 10 SHAP contributors
  - Green bars = risk-reducing features
  - Red bars = risk-increasing features
  - Horizontal layout, feature names on y-axis
  - Title: f"Why your {model_name} risk is {prediction:.0%}"

METHOD 3: `generate_health_brief(self, state: dict) -> str`
Input: current state dict from RealtimePlaybackEngine
Using template-based generation (no LLM required, but LLM-ready):
  - Analyse top risk drivers across all three models
  - Generate a 3-sentence health brief in plain English
  - Template example: "Your [metric] has been [trend] over the past [window], which is [X]% [above/below] your personal baseline. This is your biggest contributor to elevated [domain] risk today. Recommended action: [specific actionable advice]."
  - Include one concrete recommendation per risk domain that's elevated
  
METHOD 4: `generate_llm_brief(self, state: dict, api_key: str = None) -> str`
OPTIONAL — if ANTHROPIC_API_KEY is set in environment:
  - Build a structured prompt with SHAP values, personal baselines, risk scores
  - Call Claude API for a personalized, nuanced health narrative
  - Return the narrative string
  - Fall back to generate_health_brief() if API key not available
```

---

### STEP 4.2 — What-If Counterfactual Engine

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `models/counterfactual.py`.

Write a class `WhatIfSimulator` using DiCE-ml:

METHOD 1: `__init__(self, cardiac_model: CardiacRiskModel, feature_df: pd.DataFrame)`
  - Create dice_ml.Data object with feature_df and outcome_name='target'
  - Create dice_ml.Model wrapping the XGBoost pipeline
  - Create dice_ml.Dice explainer with method='random'

METHOD 2: `generate_counterfactuals(self, current_row: pd.Series, desired_risk: float = 0.3, n_cfs: int = 5) -> list[dict]`
  - Find N counterfactual scenarios where risk drops to desired_risk
  - Only allow changes to 'actionable' features: heart_rate, sleep_hours, activity_count, stress_index
  - DO NOT allow changes to: age, sex, genetic markers
  - Return list of dicts: [{'changes': {feature: new_value}, 'new_risk_score': float, 'description': str}]
  - The 'description' should be human readable: "If you slept 1.5 more hours and walked 3,000 more steps daily"

METHOD 3: `simulate_scenario(self, current_features: pd.Series, scenario: dict) -> dict`
Manual what-if: apply user-specified changes and return new risk scores
  - scenario = {'sleep_hours': 8.0, 'activity_count': 8000}
  - Modify feature row, rerun model, return {old_risk, new_risk, delta, risk_level_change}

METHOD 4: `get_optimal_action(self, current_row: pd.Series) -> dict`
Find the single smallest change that brings risk below 0.4:
  - Try each actionable feature individually
  - Return the most impactful single intervention:
    {'action': str, 'change_required': float, 'risk_reduction': float, 'difficulty': str ('easy'/'medium'/'hard')}

Difficulty heuristic:
  - sleep change > 1h = 'hard', 0.5-1h = 'medium', < 0.5h = 'easy'  
  - activity change > 3000 steps = 'hard', 1000-3000 = 'medium', < 1000 = 'easy'
```

---

---

# PHASE 5 — Streamlit Dashboard (The Demo Face)
## ⏱ Hours 28–38

---

### STEP 5.1 — Main Dashboard

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `app/streamlit_app.py` — the full Streamlit dashboard for the HealthTwin digital twin.

LAYOUT (use st.set_page_config with layout='wide'):

SIDEBAR:
  - App title: "🧬 HealthTwin — Your AI Health Twin"
  - User selector: st.selectbox("Select Subject", list of PAMAP2/WESAD subject IDs)
  - Playback speed slider: 0.1x to 10x
  - "▶ Start Live Feed" button / "⏹ Stop" button
  - "📊 Static Analysis Mode" toggle
  - Section: "What-If Simulator" with sliders:
    - Sleep hours: 4–10, step 0.5
    - Daily steps: 2000–15000, step 500
    - Stress reduction %: 0–50
    - "Simulate" button

MAIN AREA — 3 columns layout:

COLUMN 1 (30%): Health Score Radar
  - Plotly radar/spider chart with 5 axes: Cardiac, Sleep, Stress, Activity, Recovery
  - Each axis 0–100 (inverse of risk score * 100)
  - Color: green > 70, amber 40–70, red < 40
  - Update every second during live feed

COLUMN 2 (40%): Live Risk Timeline
  - Plotly line chart with 3 lines: cardiac_risk, stress_index, sleep_quality
  - Last 24 hours of data
  - Shaded bands: green (safe) / amber (watch) / red (alert) zones
  - Vertical markers for anomaly spikes
  - Updates in real-time during playback

COLUMN 3 (30%): AI Health Brief + Alerts
  - st.metric cards: Current Heart Rate, HRV, Stress Level, Sleep Score
  - Each metric shows: current value, delta from personal baseline, trend arrow
  - Below: "🧠 AI Insight" text box with generate_health_brief() output
  - Alert box (st.error/warning/success depending on risk level)

BOTTOM ROW — full width:
  - SHAP waterfall chart from shap_explainer.explain_prediction() for highest-risk domain
  - What-if simulation results (appear after user clicks Simulate)
  - Side-by-side: current risk scores vs simulated risk scores as bar chart

REAL-TIME UPDATE LOGIC:
  - Use st.empty() containers for all updating components
  - Use threading + queue to run RealtimePlaybackEngine.stream() in background
  - Main thread reads from queue and updates st.empty() containers
  - Add st.session_state management for start/stop control

Use plotly for ALL charts (not matplotlib) for interactivity.
Use st.cache_resource for model loading.
Add error handling everywhere — never let a bad data point crash the demo.
```

---

### STEP 5.2 — Demo Data Generator (Backup for Live Demo)

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `app/demo_data.py`.

Write a function `generate_demo_patient(profile: str = 'at_risk') -> pd.DataFrame` that generates 90 days of synthetic but realistic wearable data for demo purposes.

Profiles:
  'healthy': 
    - HR: 62±8, HRV: 55±12, Sleep: 7.5±0.8h, Steps: 9000±2000, Stress: low
    - Gradual improvement trend throughout 90 days
    
  'at_risk': 
    - HR: 78±12, HRV: 28±8, Sleep: 5.5±1.2h, Steps: 4500±1500, Stress: high
    - Deteriorating trend in weeks 8-10
    - Add 3 notable stress spikes (exam week pattern)
    
  'recovering':
    - Starts with at_risk profile, then improves after day 30 (simulate lifestyle change)
    - Smooth transition over 2 weeks

Each profile generates a DataFrame with columns matching the feature pipeline output.
Timestamps: every 15 minutes for 90 days.
Add realistic circadian rhythm patterns (HR lower at night, activity peaks midday).
Add weekend vs weekday patterns.
Add slight random noise with np.random.seed(42) for reproducibility.

Also write `get_demo_health_brief(profile: str) -> str` that returns a hardcoded compelling health brief for hackathon demo purposes (failsafe if model isn't trained yet).
```

---

### STEP 5.3 — Run the App

**Run:**
```bash
# From healthtwin/ root with venv active
streamlit run app/streamlit_app.py --server.port 8501 --server.headless false
```

Open `http://localhost:8501` in browser.

---

---

# PHASE 6 — Polish, Testing & Demo Prep
## ⏱ Hours 38–48

---

### STEP 6.1 — Unit Tests

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Create `tests/test_pipeline.py`.

Write pytest unit tests for:

1. test_rolling_features_shape: RollingFeatureEngine on 1000 rows of synthetic data produces expected number of output columns
2. test_baseline_normalization: PersonalBaselineNormalizer z-scores have mean≈0, std≈1 per user
3. test_risk_model_output_range: CardiacRiskModel.score_realtime always returns risk_score in [0, 1]
4. test_counterfactual_lowers_risk: WhatIfSimulator.simulate_scenario with improved sleep always reduces cardiac risk
5. test_realtime_engine_yields: RealtimePlaybackEngine.stream() yields correct schema dict
6. test_shap_explainer_keys: explain_prediction returns dict with required keys

Use pytest fixtures for model loading. Use synthetic data (no actual dataset needed).
Mock the XGBoost models where needed to speed up test runs.
All tests must run in under 30 seconds total.
```

**Run:**
```bash
pytest tests/ -v --tb=short
```

---

### STEP 6.2 — Performance Optimization

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Review `features/rolling.py` and `models/realtime_engine.py`.

Optimize for demo performance:
1. Cache rolling feature computation using functools.lru_cache where applicable
2. Precompute all 90 days of features at startup, store in st.session_state
3. During live playback, only recompute the latest window row — not the full history
4. Precompute SHAP values for the entire test dataset, cache as a numpy array
5. Add a `--precompute` flag to train_all.py that runs this pre-computation and saves to data/processed/precomputed_features.parquet

Target: dashboard update latency < 200ms per tick during live demo.
```

---

### STEP 6.3 — Demo Script (What to Say)

**[CHAT PROMPT]**

```
[PASTE CONTEXT.MD]

Write `DEMO_SCRIPT.md` — a judge-facing demo walkthrough.

Structure:
1. Hook (30 seconds): One dramatic moment — show the "at_risk" profile's risk score spike on day 60
2. The Twin Concept (60 seconds): Explain personal baseline normalization with the HR deviation example
3. Live Playback Demo (90 seconds): Start the stream, watch risk scores update in real-time
4. What-If Simulator (60 seconds): Show risk dropping when sleep slider goes from 5.5h to 7.5h
5. AI Explanation (30 seconds): Show the SHAP waterfall and health brief
6. Impact Statement (30 seconds): Preventive healthcare framing

Include talking points, likely judge questions, and your answers.
Include the 3 numbers you should memorize: model AUC, risk reduction % from what-if, and dataset size.
```

---

---

# QUICK REFERENCE — Terminal Commands

```bash
# ── Environment ──
pyenv local 3.11.9
source .venv/bin/activate
pip install -r requirements.txt

# ── Download data ──
cd data/raw && bash ../../scripts/download_data.sh

# ── Train all models ──
python -m models.train_all --models all --verbose

# ── Run feature pipeline test ──
python -m features.pipeline

# ── Run tests ──
pytest tests/ -v --tb=short

# ── Launch dashboard ──
streamlit run app/streamlit_app.py --server.port 8501

# ── Jupyter for EDA ──
jupyter lab notebooks/

# ── Check what's installed ──
pip list | grep -E "xgboost|shap|streamlit|neurokit|tsfresh"

# ── If you break something, nuke and reinstall ──
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# DATASET REFERENCE CARD

| Dataset | Domain | Size | Access | Direct URL |
|---|---|---|---|---|
| UCI Heart Disease | Cardiac risk | 303 rows | Free, instant | `archive.ics.uci.edu/dataset/45` |
| PAMAP2 | Activity + HR | 3.8GB | Free, instant | `archive.ics.uci.edu/dataset/231` |
| WESAD | Stress (ECG/EDA) | 1.7GB | Free, form | `uni-siegen.de/lab/research/wearable-sensor-data/wesad` |
| Sleep-EDF | Sleep staging | 2.5GB | Free, PhysioNet | `physionet.org/content/sleep-edfx/1.0.0/` |
| PMData | Longitudinal | 900MB | Free, form | `datasets.simula.no/pmdata/` |

**Priority for hackathon:** UCI Heart Disease (minutes) → PAMAP2 (instant) → WESAD (fill form now) → Sleep-EDF (wget partial)

---

# FOLLOW-UP PROMPTS TO SAVE

> Use these later in your build when you hit specific problems:

**When the rolling pipeline is too slow:**
```
The RollingFeatureEngine is too slow for real-time playback. Rewrite using 
numpy stride tricks and incremental updates — only recompute the window that 
changed, not the full history. Target: process one new row in < 5ms.
```

**When SHAP crashes on new data:**
```
SHAP is throwing 'feature mismatch' error. The live feature row has different 
columns than training data. Write a feature_alignment() function that takes 
any feature row and aligns it to the exact training feature schema, filling 
missing columns with 0 and dropping extra columns.
```

**When the Streamlit UI freezes:**
```
The Streamlit dashboard freezes during live playback because stream() blocks 
the main thread. Refactor using st.session_state + a threading.Thread + 
queue.Queue pattern where the background thread puts state dicts into the 
queue and the main thread drains it with st.empty().write().
```

**Last 2 hours — make it look better:**
```
The dashboard works but looks plain. Add these visual improvements to 
streamlit_app.py: (1) custom CSS for dark health-tech aesthetic using 
st.markdown with unsafe_allow_html, (2) animated gauge chart for overall 
health score using plotly.graph_objects.Indicator, (3) pulsing red border 
on the risk card when risk_level == 'Critical', (4) smooth line transitions 
by keeping last 288 points (72h) in the chart instead of 96.
```

---

*HealthTwin — built in 48 hours. Good luck.* 🧬