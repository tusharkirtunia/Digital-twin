# app/ml/health_predictor.py
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HealthTwinPredictor:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
        self.scaler = StandardScaler()
        self.explainer = None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turn raw time-series into ML features."""
        features = pd.DataFrame()
        features["avg_heart_rate"] = [df["heart_rate"].mean()]
        features["hr_variability"] = [df["heart_rate"].std()]
        features["avg_sleep"] = [df["sleep_hours"].mean()]
        features["sleep_deficit"] = [max(0, 8 - df["sleep_hours"].mean())]
        features["avg_stress"] = [df["stress_level"].mean()]
        features["avg_steps"] = [df["steps"].mean()]
        features["sedentary_days"] = [(df["steps"] < 5000).sum()]
        return features

    def train(self, df: pd.DataFrame, labels: list):
        X = self.engineer_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)
        self.explainer = shap.TreeExplainer(self.model)
        joblib.dump(self, f"models/{self.user_id}_twin.pkl")

    def predict_risk(self, df: pd.DataFrame) -> dict:
        X = self.engineer_features(df)
        X_scaled = self.scaler.transform(X)
        risk_prob = self.model.predict_proba(X_scaled)[0][1]
        
        # SHAP for explainability
        shap_values = self.explainer.shap_values(X_scaled)
        feature_names = X.columns.tolist()
        explanation = {
            feature_names[i]: round(float(shap_values[0][i]), 4)
            for i in range(len(feature_names))
        }
        return {
            "risk_score": round(float(risk_prob), 3),
            "risk_level": "high" if risk_prob > 0.7 else "medium" if risk_prob > 0.4 else "low",
            "explanation": explanation,  # which factors drove the score
        }

    def simulate_scenario(self, df: pd.DataFrame, changes: dict) -> dict:
        """What-if simulation — e.g. changes = {'avg_steps': 10000, 'avg_sleep': 8}"""
        X = self.engineer_features(df)
        for key, value in changes.items():
            if key in X.columns:
                X[key] = value
        X_scaled = self.scaler.transform(X)
        new_risk = self.model.predict_proba(X_scaled)[0][1]
        return {"simulated_risk_score": round(float(new_risk), 3)}