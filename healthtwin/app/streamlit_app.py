"""
HealthTwin — Interactive Streamlit Dashboard
============================================
The digital twin UI. Displays live metrics, risk models, SHAP explanations, 
plain-text LLM briefs, and a what-if counterfactual scenario builder.
"""

from __future__ import annotations

import time
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

# Add project root to path so 'models', 'explain', etc. can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.risk_model import CardiacRiskModel, StressModel, SleepQualityModel
from models.realtime_engine import RealtimePlaybackEngine
from explain.shap_explainer import HealthExplainer
from models.counterfactual import WhatIfSimulator
from features.pipeline import get_demo_user_df

# ===================================================================
# UI Config & State Management
# ===================================================================

st.set_page_config(
    page_title="HealthTwin Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# Authentication Guard (VAYURA Portal Integration)
# ===================================================================
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
    st.session_state.auth_user = None

# Check URL for token
q_params = st.query_params
if "token" in q_params:
    st.session_state.auth_token = q_params["token"]
    if "user" in q_params:
        st.session_state.auth_user = q_params["user"]
    
    # Optional: Clear query params to keep URL clean
    # Note: Streamlit 1.30+ clears them by reassigning, or use st.query_params.clear()
    st.query_params.clear()

if not st.session_state.auth_token:
    st.error("🔒 Access Denied. You are not authenticated.")
    st.markdown("Please log in through the VAYURA portal to access your Digital Twin dashboard.")
    st.markdown('<a href="http://localhost:8000/" target="_self"><button style="padding:10px 20px; background-color:#d95311; color:white; border:none; border-radius:5px; cursor:pointer;">Go to Login Portal</button></a>', unsafe_allow_html=True)
    st.stop()


# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .risk-high { color: #ff4d4d; font-weight: bold; }
    .risk-mod { color: #feca57; font-weight: bold; }
    .risk-low { color: #1dd1a1; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_health_system():
    """Load models, datasets, explainers, and initialize engine."""
    logger.info("Initializing HealthTwin ecosystem for Streamlit...")
    
    # 1. Load Data
    df = get_demo_user_df()
    
    # 2. Load Models
    models_dir = Path("models/saved")
    models = {}
    if (models_dir / "cardiac_latest.joblib").exists():
        models["cardiac"] = CardiacRiskModel.load(str(models_dir / "cardiac_latest.joblib"))
        models["stress"] = StressModel.load(str(models_dir / "stress_latest.joblib"))
        models["sleep"] = SleepQualityModel.load(str(models_dir / "sleep_latest.joblib"))
    else:
        st.error("Saved models not found! Please run `python -m models.train_all` first.")
        st.stop()
        
    # 3. Setup Components
    explainer = HealthExplainer(models)
    
    # Take a sample for DiCE initialization
    sim_sample = df.sample(min(100, len(df)))
    simulator = WhatIfSimulator(models["cardiac"], sim_sample)
    
    return df, models, explainer, simulator


def init_session_state(df, models):
    """Initialize Streamlit session variables."""
    if "engine" not in st.session_state:
        # Start halfway through the dataset to ensure we have rolling history
        start_idx = len(df) // 2
        st.session_state.engine = RealtimePlaybackEngine(
            df, models, speed_multiplier=2.0, start_index=start_idx
        )
        # Grab first frame
        stream_iter = st.session_state.engine.stream()
        st.session_state.stream_iter = stream_iter
        st.session_state.current_state = next(stream_iter)
        st.session_state.history = [st.session_state.current_state]
        st.session_state.is_playing = False
        st.session_state.selected_model = "cardiac"
        st.session_state.simulated_scenario = {}


df, models, explainer, simulator = load_health_system()
init_session_state(df, models)

# ===================================================================
# Sidebar: Controls & What-If
# ===================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dna-helix.png", width=60)
    st.title("HealthTwin OS")
    
    st.header("Playback Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Play" if not st.session_state.is_playing else "⏸️ Pause", use_container_width=True):
            st.session_state.is_playing = not st.session_state.is_playing
            
    with col2:
        if st.button("⏭️ Next Frame", use_container_width=True):
            st.session_state.current_state = next(st.session_state.stream_iter)
            st.session_state.history.append(st.session_state.current_state)
            
    speed = st.slider("Playback Speed", 1.0, 10.0, st.session_state.engine.speed_multiplier)
    st.session_state.engine.speed_multiplier = speed
    
    st.divider()
    
    st.header("What-If Simulator")
    st.markdown("Override vitals to see how your risk changes.")
    
    # Reset simulation
    if st.button("Reset Scenario"):
        st.session_state.simulated_scenario = {}
        st.rerun()

    raw = st.session_state.current_state["raw_signals"]
    
    sim_hr = st.number_input("Heart Rate", value=st.session_state.simulated_scenario.get("heart_rate", raw.get("heart_rate", 70.0)))
    sim_hrv = st.number_input("HRV (RMSSD)", value=st.session_state.simulated_scenario.get("hrv_rmssd", raw.get("hrv_rmssd", 40.0)))
    sim_sleep = st.number_input("Sleep Hours", value=st.session_state.simulated_scenario.get("sleep_hours", raw.get("sleep_hours", 7.0)))
    
    if st.button("Apply Scenario", type="primary"):
        st.session_state.simulated_scenario = {
            "heart_rate": sim_hr,
            "hrv_rmssd": sim_hrv,
            "sleep_hours": sim_sleep
        }
        
    st.divider()
    
    st.header("Optimal Action Engine")
    if st.button("Get AI Suggestion"):
        cf_df = pd.DataFrame([st.session_state.current_state["full_feature_row"]])
        action = simulator.get_optimal_action(cf_df.iloc[0])
        st.info(action["msg"])
        st.metric("Risk Reduction", f"{(action['risk_reduction']*100):.1f}%")

# ===================================================================
# Main Dashboard
# ===================================================================

# Get active state
state = st.session_state.current_state
raw = state["raw_signals"]
risks = state["risk_scores"]
ts = state["timestamp"]

st.title(f"Live Dashboard — {ts.strftime('%Y-%m-%d %H:%M')}")

# -- Subheader: Top Level Vitals --
col_hr, col_hrv, col_slp, col_act = st.columns(4)

col_hr.metric("Heart Rate", f"{raw.get('heart_rate', 0):.0f} bpm")
col_hrv.metric("HRV", f"{raw.get('hrv_rmssd', 0):.0f} ms")
col_slp.metric("Sleep", f"{raw.get('sleep_hours', 0):.1f} h")
col_act.metric("Activity", f"{raw.get('activity_count', 0):.0f}")

st.markdown("---")

# -- Middle: Models & Explainability --
left_col, right_col = st.columns([1, 1.5])

with left_col:
    st.subheader("Vitals Risk Assessment")
    
    tab_cardiac, tab_stress, tab_sleep = st.tabs(["Cardiac", "Stress", "Sleep"])
    
    active_tab_model = "cardiac" # We will sync this based on clicks in advanced streamlit, for now static loops
    
    # Render Risk Cards
    def render_risk(model_name, tab):
        with tab:
            info = risks[model_name]
            score = info["risk_score"]
            level = info["risk_level"]
            
            # Check simulation override
            if st.session_state.simulated_scenario:
                cf_df = pd.DataFrame([st.session_state.current_state["full_feature_row"]])
                sim_res = simulator.simulate_scenario(cf_df.iloc[0], st.session_state.simulated_scenario)
                st.markdown(f"### Simulated Risk: **{sim_res['new_risk']:.0%}** ({sim_res['new_level']})")
                st.markdown(f"*Original: {score:.0%}*  |  Delta: {sim_res['delta']:+.2%}")
            else:
                color = "risk-low" if level == "Low" else "risk-mod" if level == "Moderate" else "risk-high"
                st.markdown(f"<h1 class='{color}'>{score:.0%} ({level})</h1>", unsafe_allow_html=True)
                
            st.markdown(f"**Top Risk Drivers:** {', '.join([f.replace('_', ' ').title() for f in info['top_risk_factors'][:3]])}")

    render_risk("cardiac", tab_cardiac)
    render_risk("stress", tab_stress)
    render_risk("sleep", tab_sleep)

    st.subheader("AI Health Brief")
    with st.spinner("Generating brief..."):
        # Real-time brief generation
        brief = explainer.generate_llm_brief(state)
        st.info(brief)

with right_col:
    st.subheader("SHAP Driver Analysis")
    
    # Allow user to select which model to explain
    explain_target = st.selectbox("Explain Prediction For:", ["cardiac", "stress", "sleep"])
    
    cf_df = pd.DataFrame([st.session_state.current_state["full_feature_row"]])
    current_row = cf_df.iloc[0]
    
    if st.session_state.simulated_scenario:
        # Override features for explanation if simulated
        override_row = current_row.copy()
        for k, v in st.session_state.simulated_scenario.items():
            if k in override_row.index:
                override_row[k] = v
        current_row = override_row
        st.caption("*Showing explanations for simulated scenario")

    try:
        exp_res = explainer.explain_prediction(explain_target, current_row)
        st.plotly_chart(exp_res["plotly_waterfall_fig"], use_container_width=True)
    except Exception as e:
        st.error(f"Explanation failed: {e}")

# ===================================================================
# Bottom: Timeseries
# ===================================================================
st.markdown("---")
st.subheader("Risk History (Past 24h)")

# Convert history to DataFrame for easy Plotly mapping
if len(st.session_state.history) > 0:
    hist_df = pd.DataFrame([{
        "timestamp": s["timestamp"],
        "Cardiac": s["risk_scores"]["cardiac"]["risk_score"],
        "Stress": s["risk_scores"]["stress"]["risk_score"],
        "Sleep": s["risk_scores"]["sleep"]["risk_score"]
    } for s in st.session_state.history[-96:]]) # Last 24 hours (96 @ 15m)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["Cardiac"], name="Cardiac", line=dict(color="#ff4d4d")))
    fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["Stress"], name="Stress", line=dict(color="#feca57")))
    fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["Sleep"], name="Sleep", line=dict(color="#54a0ff")))
    
    fig.update_layout(
        template="plotly_dark", 
        yaxis_title="Risk Score", 
        yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# Engine Loop Tick
# ===================================================================
if st.session_state.is_playing:
    time.sleep(1.0) # Slight delay to avoid completely overwhelming UI
    # Grab next state
    try:
        st.session_state.current_state = next(st.session_state.stream_iter)
        st.session_state.history.append(st.session_state.current_state)
        # Keep history bounded
        if len(st.session_state.history) > 1000:
            st.session_state.history = st.session_state.history[-1000:]
        st.rerun()
    except StopIteration:
        st.session_state.is_playing = False
        st.warning("End of dataset reached.")
