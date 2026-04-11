# app/api/routes/health.py
from fastapi import APIRouter, Depends, HTTPException
from app.services.wearable_ingest import ingest_wearable_data, get_user_timeseries

from app.core.auth import get_current_user
import joblib, os

router = APIRouter(prefix="/health", tags=["health"])

@router.post("/ingest")
async def ingest_data(data: dict, current_user=Depends(get_current_user)):
    """Receive wearable data from device/app."""
    ingest_wearable_data(str(current_user.id), data)
    return {"status": "ok"}

@router.get("/predict/{risk_type}")
async def predict_risk(risk_type: str, current_user=Depends(get_current_user)):
    """
    risk_type: "heart_disease" | "stress" | "sleep_disorder"
    Returns risk score + SHAP explanation.
    """
    df = get_user_timeseries(str(current_user.id), days=30)
    if df.empty:
        raise HTTPException(status_code=404, detail="Not enough data yet")
    
    model_path = f"models/{current_user.id}_{risk_type}.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Twin not trained yet")
    
    
    
    

@router.post("/simulate")
async def simulate_whatif(payload: dict, current_user=Depends(get_current_user)):
    """
    payload = {
        "risk_type": "heart_disease",
        "changes": {"avg_steps": 10000, "avg_sleep": 8.0}
    }
    """
    df = get_user_timeseries(str(current_user.id), days=30)
    model_path = f"models/{current_user.id}_{payload['risk_type']}.pkl"

    
    

@router.get("/insights")
async def get_insights(current_user=Depends(get_current_user)):
    """Return top 3 AI-generated health insights for the user."""
    df = get_user_timeseries(str(current_user.id), days=7)
    insights = []
    if df["sleep_hours"].mean() < 6:
        insights.append({"type": "warning", "message": "Avg sleep below 6h — risk of stress elevated"})
    if df["steps"].mean() < 5000:
        insights.append({"type": "tip", "message": "Low activity detected. 8000+ steps reduces heart risk by 18%"})
    if df["heart_rate"].std() > 15:
        insights.append({"type": "alert", "message": "High heart rate variability — consult a physician"})
    return {"insights": insights}