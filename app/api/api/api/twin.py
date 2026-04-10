from fastapi import APIRouter, Depends
from app.core.auth import get_current_user
from app.workers.tasks import train_user_twin

router = APIRouter(prefix="/twin", tags=["twin"])

@router.post("/train/{risk_type}")
def trigger_training(risk_type: str, current_user=Depends(get_current_user)):
    """Kick off background model training for this user."""
    task = train_user_twin.delay(str(current_user), risk_type)
    return {"task_id": task.id, "status": "training started"}

@router.get("/status/{task_id}")
def check_training_status(task_id: str):
    from app.workers.tasks import celery_app
    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result}