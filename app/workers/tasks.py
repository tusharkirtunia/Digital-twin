# app/workers/tasks.py
from celery import Celery
from app.services.wearable_ingest import get_user_timeseries
from app.ml.health_predictor import HealthTwinPredictor
import numpy as np

celery_app = Celery("health_twin", broker="amqp://guest@localhost//", backend="redis://localhost:6379/0")

@celery_app.task
def train_user_twin(user_id: str, risk_type: str):
    """Triggered when enough data is collected. Trains a personal model."""
    df = get_user_timeseries(user_id, days=90)
    if len(df) < 30:
        return {"status": "insufficient_data"}
    
    # For demo: generate synthetic labels based on thresholds
    labels = (
        (df["heart_rate"] > 100).astype(int) |
        (df["sleep_hours"] < 5).astype(int) |
        (df["stress_level"] > 7).astype(int)
    ).tolist()

    predictor = HealthTwinPredictor(user_id)
    predictor.train(df, labels)
    return {"status": "trained", "user_id": user_id, "risk_type": risk_type}

@celery_app.task
def generate_weekly_report(user_id: str):
    """Runs every Monday — generates a health summary."""
    df = get_user_timeseries(user_id, days=7)
    report = {
        "avg_heart_rate": round(df["heart_rate"].mean(), 1),
        "avg_sleep": round(df["sleep_hours"].mean(), 1),
        "total_steps": int(df["steps"].sum()),
        "avg_stress": round(df["stress_level"].mean(), 1),
    }
    # TODO: save report to DB and send email/push notification
    return report