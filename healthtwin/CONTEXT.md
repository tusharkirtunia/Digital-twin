# HealthTwin — Project Context

## PROJECT STACK
- Language: Python 3.11
- Environment: pyenv + venv, macOS
- UI: Streamlit
- ML: XGBoost, scikit-learn
- XAI: SHAP, DiCE-ml
- Feature engineering: tsfresh, neurokit2, pandas rolling windows
- Physiology: neurokit2 for ECG/HRV processing
- Visualization: Plotly, matplotlib
- LLM: OpenAI GPT-4o-mini for health briefs

## FOLDER STRUCTURE
```
healthtwin/
├── data/raw/          # downloaded datasets
├── data/processed/    # cleaned, merged, normalized + loaders.py
├── features/          # rolling.py, baseline.py, hrv.py, pipeline.py
├── models/            # risk_model.py, train_all.py, realtime_engine.py, counterfactual.py
├── explain/           # shap_explainer.py
├── app/               # streamlit_app.py, demo_data.py
├── notebooks/         # eda.ipynb
├── tests/             # unit tests
└── CONTEXT.md
```

## DATASETS IN USE
1. UCI Heart Disease (Cleveland) — cardiac risk factors
2. WESAD — wearable stress/affect detection (ECG, EDA, TEMP, ACC)
3. PAMAP2 — physical activity monitoring with IMU + HR
4. Sleep-EDF (PhysioNet) — sleep stage classification

## CORE CONCEPTS
- Rolling time-series features: 24h, 7d, 30d windows with statistical aggregates
- Personal baseline normalization: z-score each metric against user's own rolling mean/std
- Simulated real-time playback: chronological row-by-row replay with time.sleep() in Streamlit
- What-if simulator: DiCE counterfactual engine on trained XGBoost model
- XAI: SHAP waterfall charts + GPT-4o-mini plain-English health brief

## CURRENT PHASE: Phase 1 — Environment & Data Pipeline
## LAST FILE MODIFIED: data/processed/loaders.py
