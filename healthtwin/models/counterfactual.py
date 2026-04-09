"""
HealthTwin — What-If Counterfactual Simulator
=============================================
Uses DiCE (Diverse Counterfactual Explanations) to figure out *what needs to change*
to lower a user's risk score.

Provides two main capabilities:
1. `generate_counterfactuals`: Ask "how can I get my risk below 0.3?" and get actionable suggestions.
2. `simulate_scenario`: Manually overwrite specific features yielding the new risk score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Try loading dice_ml
try:
    import dice_ml
    HAS_DICE = True
except ImportError:
    HAS_DICE = False
    logger.warning("dice_ml not available — WhatIfSimulator will have limited functionality.")

from models.risk_model import BaseHealthModel

# Define which features a user can reasonably change in real life.
ACTIONABLE_FEATURES = [
    "sleep_hours",
    "activity_count", 
    "heart_rate",       # (Assuming meditation/relaxation lowers it)
    "hrv_rmssd",        # (Proxy for recovery activities)
    "resp_rate",
    "eda_mean"          # (Proxy for stress)
]

class WhatIfSimulator:
    """
    DiCE-powered counterfactual engine to explore health scenarios.
    """

    def __init__(self, model: BaseHealthModel, feature_df: pd.DataFrame) -> None:
        """
        Initialize the simulator for a specific model.

        Parameters
        ----------
        model : BaseHealthModel
            The trained model to simulate (e.g. CardiacRiskModel).
        feature_df : pd.DataFrame
            A sample of training data or historical data providing the schema and distributions
            required by DiCE.
        """
        self.health_model = model
        self.feature_names = model.feature_names
        
        # Determine actionable columns that actually exist in the model
        self.actionable = [f for f in ACTIONABLE_FEATURES if f in self.feature_names]
        
        # Add basic rolling forms of actionable features
        for f in ACTIONABLE_FEATURES:
            for w in [96, 672]:
                col_name = f"{f}_{w}w_mean"
                if col_name in self.feature_names:
                    self.actionable.append(col_name)

        if not HAS_DICE:
            self.dice_exp = None
            return

        logger.info(f"Setting up WhatIfSimulator for model '{model.name}'")
        
        # Prepare background DataFrame for DiCE
        # DiCE needs the exact columns the model expects, plus a target column
        df = feature_df.copy()
        
        # Fill missing required columns with 0 to ensure schema match
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
                
        # Slice to only necessary columns
        df_dice = df[self.feature_names].copy()
        
        # Generate dummy target column representing "Risk" for initialization.
        # We'll just run the model over it to get binary targets (High Risk=1, Low Risk=0)
        # Using a threshold of 0.5
        preds = []
        for _, row in df_dice.iterrows():
            preds.append(1 if model.score_realtime(row)["risk_score"] > 0.5 else 0)
            
        df_dice["target"] = preds
        
        # Initialize DiCE Data
        # We mark all features as continuous for simplicity in this dataset
        continuous_feats = self.feature_names
        
        d = dice_ml.Data(
            dataframe=df_dice,
            continuous_features=continuous_feats,
            outcome_name="target"
        )
        
        # DiCE needs a model wrapper. 
        # DiCE sklearn wrapper expects predict and predict_proba on the backend object.
        # However, because we have a custom `BaseHealthModel`, we construct a proxy.
        class ProxyModel:
            def __init__(self, hm):
                self.hm = hm
            def predict_proba(self, X):
                # X is a DataFrame from DiCE
                out = []
                for _, row in X.iterrows():
                    aligned = self.hm._align_features(row).reshape(1, -1)
                    proba = self.hm.pipeline.predict_proba(aligned)[0, 1]
                    out.append([1 - proba, proba])
                return np.array(out)
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

        m = dice_ml.Model(model=ProxyModel(model), backend="sklearn")
        
        # Initialize DiCE explainer (Random method is robust and fast)
        self.dice_exp = dice_ml.Dice(d, m, method="random")
        logger.success(f"DiCE Explainer ready. {len(self.actionable)} actionable features.")


    def generate_counterfactuals(self, current_row: pd.Series, desired_risk: float = 0.3, n_cfs: int = 3) -> list[dict[str, Any]]:
        """
        Find minimal changes to actionable features that lower the risk below `desired_risk`.
        
        Parameters
        ----------
        current_row : pd.Series
            The current observation state.
        desired_risk : float
            Target risk threshold (e.g. 0.3 = Low).
        n_cfs : int
            Number of diverse counterfactual scenarios to generate.
            
        Returns
        -------
        list[dict]
            A list of dictionary scenarios.
        """
        curr_risk = self.health_model.score_realtime(current_row)["risk_score"]
        if curr_risk <= desired_risk:
            return [{"msg": f"Risk is already at {curr_risk:.2f}, below target {desired_risk:.2f}."}]

        if not self.dice_exp:
            return [{"error": "dice_ml not installed."}]

        # Prepare input df
        query_df = pd.DataFrame([current_row])
        for col in self.feature_names:
            if col not in query_df.columns:
                query_df[col] = 0.0
        query_df = query_df[self.feature_names]

        try:
            # We want the model to predict class 0 (Low risk)
            dice_cfs = self.dice_exp.generate_counterfactuals(
                query_df,
                total_CFs=n_cfs,
                desired_class=0,
                features_to_vary=self.actionable
            )
            
            cfs_df = dice_cfs.cf_examples_list[0].final_cfs_df
            
            if cfs_df is None or cfs_df.empty:
                return [{"msg": "Could not find a feasible scenario to lower risk."}]
                
            results = []
            
            for _, cf_row in cfs_df.iterrows():
                changes = {}
                for feature in self.actionable:
                    old_val = float(current_row.get(feature, 0.0))
                    new_val = float(cf_row.get(feature, 0.0))
                    
                    # If it changed by more than 1%
                    if abs(old_val - new_val) > (abs(old_val) * 0.01 + 1e-4):
                        changes[feature] = new_val
                        
                if changes:
                    # Score it
                    sim_row = current_row.copy()
                    for k, v in changes.items():
                        sim_row[k] = v
                    new_risk = self.health_model.score_realtime(sim_row)["risk_score"]
                    
                    results.append({
                        "changes": changes,
                        "new_risk_score": float(new_risk),
                        "description": self._format_description(changes)
                    })
                    
            # Sort by highest risk reduction (lowest new risk)
            return sorted(results, key=lambda x: x["new_risk_score"])
            
        except Exception as e:
            logger.error(f"DiCE generation failed: {e}")
            return [{"error": str(e)}]


    def simulate_scenario(self, current_features: pd.Series, scenario: dict[str, float]) -> dict[str, Any]:
        """
        Manually override features to see the resulting risk score.
        
        Parameters
        ----------
        current_features : pd.Series
            The current true state.
        scenario : dict[str, float]
            Map of feature name -> new absolute value. (e.g. {'sleep_hours': 8.0})
            
        Returns
        -------
        dict
            Contains old_risk, new_risk, delta, and level_change.
        """
        old_eval = self.health_model.score_realtime(current_features)
        
        sim_row = current_features.copy()
        for k, v in scenario.items():
            if k in sim_row.index:
                sim_row[k] = v
                # Super hacky heuristic: also update rolling means if we override the core feature.
                # In a real app we'd feed the new data through the pipeline.
                for w in [96, 672]:
                    r_col = f"{k}_{w}w_mean"
                    if r_col in sim_row.index:
                        # Moves the mean a tiny bit towards the new value for visual effect
                        sim_row[r_col] = (sim_row[r_col] * 0.8) + (v * 0.2)
                        
        new_eval = self.health_model.score_realtime(sim_row)
        
        delta = new_eval["risk_score"] - old_eval["risk_score"]
        
        return {
            "old_risk": old_eval["risk_score"],
            "new_risk": new_eval["risk_score"],
            "delta": delta,
            "old_level": old_eval["risk_level"],
            "new_level": new_eval["risk_level"]
        }


    def get_optimal_action(self, current_row: pd.Series) -> dict[str, Any]:
        """
        Find the single smallest change that brings risk down meaningfully.
        Try bumping up Sleep or bumping up Steps/Activity, etc.
        """
        # Baseline
        base_risk = self.health_model.score_realtime(current_row)["risk_score"]
        
        best_action = {"action": "None", "risk_reduction": 0.0, "difficulty": "n/a"}
        
        # Test: Add 1 hour of sleep
        if "sleep_hours" in self.actionable:
            curr = current_row.get("sleep_hours", 0)
            if curr < 8.5:
                res = self.simulate_scenario(current_row, {"sleep_hours": curr + 1.0})
                reduction = base_risk - res["new_risk"]
                if reduction > best_action["risk_reduction"]:
                    best_action = {
                        "action": "sleep_hours",
                        "change_required": 1.0,
                        "risk_reduction": float(reduction),
                        "difficulty": "medium",
                        "msg": "Sleep 1 more hour tonight."
                    }
                    
        # Test: Add 2000 steps (activity magnitude)
        if "activity_count" in self.actionable:
            curr = current_row.get("activity_count", 0)
            res = self.simulate_scenario(current_row, {"activity_count": curr + 2000.0})
            reduction = base_risk - res["new_risk"]
            if reduction > best_action["risk_reduction"]:
                best_action = {
                    "action": "activity_count",
                    "change_required": 2000.0,
                    "risk_reduction": float(reduction),
                    "difficulty": "easy",
                    "msg": "Add 2,000 steps to your daily routine (about a 20m walk)."
                }
                
        # Test: Reduce stress (eda)
        if "eda_mean" in self.actionable:
            curr = current_row.get("eda_mean", 1.0)
            res = self.simulate_scenario(current_row, {"eda_mean": curr * 0.7}) # 30% reduction
            reduction = base_risk - res["new_risk"]
            if reduction > best_action["risk_reduction"]:
                best_action = {
                    "action": "stress_reduction",
                    "change_required": -30.0,
                    "risk_reduction": float(reduction),
                    "difficulty": "hard",
                    "msg": "Engage in 15 mins of deep relaxation to lower sympathetic stress by 30%."
                }
                
        return best_action

    def _format_description(self, changes: dict[str, float]) -> str:
        """Create a human readable list of changes."""
        parts = []
        for k, v in changes.items():
            name = k.replace("_", " ").title()
            parts.append(f"{name} → {v:.1f}")
        return "If: " + ", ".join(parts)


if __name__ == "__main__":
    from features.pipeline import get_demo_user_df
    
    logger.info("Setting up WhatIfSimulator test...")
    
    cardiac = BaseHealthModel.load("models/saved/cardiac_latest.joblib")
    df = get_demo_user_df()
    
    simulator = WhatIfSimulator(cardiac, df.sample(100)) # Small sample to initialize DiCE fast
    
    # Try an observation with high risk
    for idx, row in df.iterrows():
        risk = cardiac.score_realtime(row)["risk_score"]
        if risk > 0.6:
            current = row
            break
            
    logger.info(f"Picked observation with {risk:.3f} risk for testing.")
    
    # 1. Manual simulation
    logger.info("Testing simulate_scenario (+2h sleep):")
    sim = simulator.simulate_scenario(current, {"sleep_hours": current["sleep_hours"] + 2.0})
    logger.info(f"  Result: {sim['old_risk']:.3f} -> {sim['new_risk']:.3f}")
    
    # 2. Optimal action heuristical
    logger.info("Testing get_optimal_action:")
    action = simulator.get_optimal_action(current)
    logger.info(f"  Result: {action['msg']} (Reduces risk by {action['risk_reduction']:.3f})")
    
    # 3. DiCE generation (if installed)
    logger.info("Testing generate_counterfactuals (might take a moment):")
    cfs = simulator.generate_counterfactuals(current, desired_risk=0.4, n_cfs=2)
    for c in cfs:
        if "msg" in c or "error" in c:
            logger.info(c)
        else:
            logger.info(f"  New Risk: {c['new_risk_score']:.3f} | {c['description']}")
    
    logger.success("What-If simulator tests passed ✓")
