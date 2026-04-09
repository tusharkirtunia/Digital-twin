"""
HealthTwin — Explainability via SHAP & LLM
============================================
Provides local explainability for model predictions using SHAP (SHapley Additive exPlanations)
and generates human-readable health briefs (via templates or OpenAI LLM).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from models.risk_model import BaseHealthModel

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

class HealthExplainer:
    """
    SHAP-based explainer for HealthTwin models + Health Brief generation.
    """

    def __init__(self, models: dict[str, BaseHealthModel]) -> None:
        """
        Initialize the explainer with trained models.

        Parameters
        ----------
        models : dict[str, BaseHealthModel]
            Dictionary of models to explain, e.g. {'cardiac': model_c, 'stress': model_s}
        """
        self.models = models
        self.explainers: dict[str, shap.TreeExplainer] = {}
        self.feature_names: dict[str, list[str]] = {}

        logger.info(f"Initializing HealthExplainer for models: {list(models.keys())}")

        for name, model in models.items():
            if not model.is_trained or model.pipeline is None:
                logger.warning(f"Model '{name}' is not trained. Skipping SHAP init.")
                continue

            try:
                # XGBoost wrapper is the final step in the pipeline
                xgb_model = model.pipeline.named_steps["model"]
                self.explainers[name] = shap.TreeExplainer(xgb_model)
                self.feature_names[name] = model.feature_names
            except Exception as e:
                logger.error(f"Failed to initialize TreeExplainer for {name}: {e}")

    def explain_prediction(self, model_name: str, feature_row: pd.Series) -> dict[str, Any]:
        """
        Explain a single prediction using SHAP.

        Returns
        -------
        dict
            Contains shap_values, base_value, prediction, top drivers (pos/neg),
            and a Plotly waterfall figure.
        """
        if model_name not in self.explainers:
            raise ValueError(f"No SHAP explainer available for model '{model_name}'.")

        model = self.models[model_name]
        explainer = self.explainers[model_name]
        fnames = self.feature_names[model_name]

        # 1. Align features and apply preprocessing manually to feed exactly into XGBoost
        aligned = model._align_features(feature_row).reshape(1, -1)

        # Apply Imputer & Scaler
        pipeline = model.pipeline
        if pipeline is None:
            raise ValueError("Pipeline is empty.")

        # Find the index of the model step
        model_step_idx = list(pipeline.named_steps.keys()).index("model")

        # Transform using the steps BEFORE the model
        X_trans = aligned
        for i in range(model_step_idx):
            step = pipeline.steps[i][1]
            if hasattr(step, "transform"):
                X_trans = step.transform(X_trans)

        # 2. Get SHAP values (TreeExplainer on log-odds margin)
        # For predicting probability we would need a proper link function, 
        # but displaying log-odds impacts in waterfall is standard.
        shap_values_obj = explainer(X_trans)
        
        # Typically shape is (1, n_features). If multi-class, could be (1, n_features, n_classes)
        # We index 0 to get the 1D array for this sample
        sv = shap_values_obj.values[0]
        bv = shap_values_obj.base_values[0]
        
        # In case of Binary classification, SHAP tree explainer might return shape (n_features, )
        # or (n_features, 2). Assuming positive class is last dimension if 2D.
        if len(sv.shape) > 1:
            sv = sv[:, 1]
            if isinstance(bv, (list, np.ndarray)):
                bv = bv[1]

        # Get final probability from the pipeline (0 to 1)
        # (For sleep model, risk was inverted in score_realtime, but SHAP here explains the base model output.
        # We'll just explain raw model output here to keep SHAP logic sound)
        pred_proba = pipeline.predict_proba(aligned)[0, 1]
        
        # The base model for Sleep Quality predicts "Good Sleep" (1=Good, 0=Bad)
        # The SHAP values will be explaining probability of GOOD sleep.
        # So "risk-increasing" means dropping the probability.

        # 3. Compile dictionary of impact
        impact_dict = {feat: val for feat, val in zip(fnames, sv)}

        # Sort features by absolute SHAP value
        sorted_feats = sorted(fnames, key=lambda f: abs(impact_dict[f]), reverse=True)

        # Identify drivers
        # Postive driver = pushes prediction up. Negative = pushes prediction down.
        drivers_pos = []
        drivers_neg = []
        
        for f in sorted_feats:
            val = impact_dict[f]
            # Ensure we safely grab the original value, fallback 0.0
            feat_val = feature_row.get(f, 0.0)
            if pd.isna(feat_val):
                feat_val = 0.0
                
            entry = (f, feat_val, val)
            
            if val > 0:
                drivers_pos.append(entry)
            else:
                drivers_neg.append(entry)

        top_pos = drivers_pos[:5]
        top_neg = drivers_neg[:5]

        # 4. Generate Plotly Waterfall Figure (Top 10 total)
        top_10 = sorted_feats[:10]
        top_10_vals = [impact_dict[f] for f in top_10]
        top_10_names = top_10

        # Create human-readable names with feature values
        y_labels = []
        for f in top_10:
            v_val = feature_row.get(f, 0.0)
            if pd.isna(v_val):
                v_val = 0.0
            
            # Format nicely
            if abs(v_val) < 0.01:
                label = f"{f} ({v_val:.4f})"
            elif abs(v_val) > 1000:
                label = f"{f} ({v_val:.0f})"
            else:
                label = f"{f} ({v_val:.2f})"
            y_labels.append(label)

        # Reverse for bottom-to-top plotting
        y_labels = y_labels[::-1]
        top_10_vals = top_10_vals[::-1]

        # Base text: "Why your X risk is Y"
        # Adjust logic for sleep vs others
        display_pred = pred_proba
        title_str = f"{model_name.capitalize()} Risk Prediction Driver Impact"
        if model_name == "sleep":
            display_pred = 1.0 - pred_proba
            title_str = "Sleep Quality Risk Driver Impact (Inverted)"

        colors = ['#2ca02c' if (v < 0 and model_name != "sleep") or (v > 0 and model_name == "sleep") else '#d62728' for v in top_10_vals]

        fig = go.Figure(go.Bar(
            x=top_10_vals,
            y=y_labels,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.2f}" for v in top_10_vals],
            textposition="auto"
        ))
        
        # Green = good (reduces risk), Red = bad (increases risk)
        # If cardiac/stress, neg SHAP = decreases prob of disease = Green
        # If sleep, pos SHAP = increases prob of GOOD sleep = Green
        
        fig.update_layout(
            title=f"{title_str} (Score: {display_pred:.0%})",
            xaxis_title="SHAP Value (Impact on Log-Odds)",
            yaxis_title="Feature (Current Value)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            height=400,
        )

        return {
            "shap_values": impact_dict,
            "base_value": float(bv),
            "prediction": float(display_pred),
            "top_drivers_positive": top_pos,
            "top_drivers_negative": top_neg,
            "plotly_waterfall_fig": fig
        }

    def generate_health_brief(self, state: dict[str, Any]) -> str:
        """
        Generate a plain-English health brief using a template algorithm.

        Parameters
        ----------
        state : dict
            Current state from RealtimePlaybackEngine.

        Returns
        -------
        str
            A 2-3 sentence markdown health brief.
        """
        # Find the domain with the highest risk
        highest_domain = None
        max_risk = -1.0
        
        for domain, info in state.get("risk_scores", {}).items():
            if info["risk_score"] > max_risk:
                max_risk = info["risk_score"]
                highest_domain = domain
                
        if not highest_domain or max_risk < 0.3:
            return "✅ **All systems stable.** Your metrics are currently within your personal baseline, resulting in low overall risk. Keep up the good routines!"

        # Extract top feature contributing to that risk
        domain_info = state["risk_scores"][highest_domain]
        top_factors = domain_info.get("top_risk_factors", [])
        
        if not top_factors:
            return f"⚠️ **Elevated {highest_domain} risk detected** ({max_risk:.0%} risk score). Please monitor your levels closely."

        primary_factor = top_factors[0]
        
        # Check normalization (if it's above/below baseline)
        zscore_key = f"{primary_factor}_personal_zscore"
        zscore = state.get("personal_zscores", {}).get(zscore_key, 0.0)
        
        direction_str = "higher" if zscore > 0 else "lower"
        abs_z = abs(zscore)
        severity = "significantly " if abs_z > 2.0 else ""
        
        # Simplify formatting of the factor name
        clean_factor = primary_factor.replace("_", " ").replace("rmssd", "HRV").title()

        brief = (
            f"⚠️ **Elevated {highest_domain.title()} Risk ({max_risk:.0%}):** "
            f"Your {clean_factor} is {severity}{direction_str} than your personal baseline (Z-score: {zscore:+.1f}). "
            f"This is the biggest contributor to your elevated {highest_domain} risk today. "
        )
        
        # Add a simple recommendation based on the primary factor
        if "sleep" in primary_factor.lower():
            brief += "\n\n💡 **Action:** Prioritize sleep hygiene tonight. Aiming for an extra hour could significantly restore your baseline."
        elif "stress" in highest_domain or "eda" in primary_factor or "hrv" in primary_factor.lower():
            brief += "\n\n💡 **Action:** Consider taking a 5-minute deep breathing or meditation break right now to balance your nervous system."
        elif "activity" in primary_factor.lower() or "step" in primary_factor.lower():
            brief += "\n\n💡 **Action:** A brisk 15-minute walk could help stabilize this metric."
        else:
            brief += "\n\n💡 **Action:** Monitor this trend and avoid intense exertion for the next few hours."

        return brief

    def generate_llm_brief(self, state: dict[str, Any], api_key: Optional[str] = None) -> str:
        """
        Generate a rich health brief using OpenAI's GPT-4o-mini model.

        Falls back to generate_health_brief if API key is missing or call fails.
        """
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            logger.debug("No OpenAI API key found, falling back to template brief.")
            return self.generate_health_brief(state)

        # Extract core summary state to feed the LLM
        # Keep it small to save tokens
        risk_summary = {k: f"{v['risk_score']:.2f} ({v['risk_level']})" for k, v in state.get("risk_scores", {}).items()}
        
        zscores = state.get("personal_zscores", {})
        # Only take extreme z-scores > 1.5
        extreme_zscores = {k.replace("_personal_zscore", ""): round(v, 2) for k, v in zscores.items() if abs(v) > 1.5}
        
        # Basic raw stats
        raw = state.get("raw_signals", {})
        basics = {
            "HR": round(raw.get("heart_rate", 0), 1),
            "HRV": round(raw.get("hrv_rmssd", 0), 1),
            "Sleep": round(raw.get("sleep_hours", 0), 1),
        }

        system_prompt = (
            "You are HealthTwin, an AI medical and wellbeing assistant analyzing a patient's digital twin data. "
            "You will be given a JSON summary of their current state. "
            "Your job is to write a single paragraph (max 3-4 sentences) explaining their current health status. "
            "Focus on the highest risk domain. Mention if metrics are notably above/below their personal baseline (measured in Z-scores). "
            "Conclude with exactly one concrete, immediate, actionable recommendation. "
            "Keep the tone encouraging, objective, and professional."
        )

        user_prompt = f"Current State:\n- Raw Vitals: {basics}\n- Risks: {risk_summary}\n- Notable Baseline Deviations (Z-scores): {extreme_zscores}"

        try:
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.4,
            )
            llm_text = response.choices[0].message.content.strip()
            return f"🧠 **AI Health Insight**\n\n{llm_text}"
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self.generate_health_brief(state)

# ===================================================================
# Self-test
# ===================================================================
if __name__ == "__main__":
    from features.pipeline import get_demo_user_df
    
    logger.info("Initializing Explainer test...")
    
    from models.risk_model import CardiacRiskModel, StressModel, SleepQualityModel
    
    # We must load trained models, or train mock ones.
    # Let's assume models/saved/cardiac_latest.joblib exists.
    cardiac = CardiacRiskModel.load("models/saved/cardiac_latest.joblib")
    stress = StressModel.load("models/saved/stress_latest.joblib")
    sleep = SleepQualityModel.load("models/saved/sleep_latest.joblib")
    
    models = {"cardiac": cardiac, "stress": stress, "sleep": sleep}
    
    explainer = HealthExplainer(models)
    
    df = get_demo_user_df()
    sample_row = df.iloc[-1] # take last row
    
    logger.info("Testing SHAP explanation for Cardiac model...")
    exp = explainer.explain_prediction("cardiac", sample_row)
    
    logger.info(f"Base Value: {exp['base_value']:.4f}")
    logger.info(f"Top 3 Positive Drivers: {exp['top_drivers_positive'][:3]}")
    logger.info(f"Top 3 Negative Drivers: {exp['top_drivers_negative'][:3]}")
    
    # Mock state
    mock_state = {
        "risk_scores": {"cardiac": {"risk_score": 0.8, "risk_level": "High", "top_risk_factors": ["hrv_rmssd", "sleep_hours"]}},
        "personal_zscores": {"hrv_rmssd_personal_zscore": -2.5, "sleep_hours_personal_zscore": -1.8},
        "raw_signals": {"hrv_rmssd": 15, "sleep_hours": 4.5, "heart_rate": 85}
    }
    
    logger.info("Testing Template Brief:")
    print(explainer.generate_health_brief(mock_state))
    
    logger.info("Testing LLM Brief (will fallback if no key):")
    print(explainer.generate_llm_brief(mock_state))
    
    logger.success("Explainer module tests completed.")
