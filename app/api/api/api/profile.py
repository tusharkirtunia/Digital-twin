from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import HealthProfile
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/profile", tags=["profile"])

class ProfileRequest(BaseModel):
    age: float
    weight_kg: float
    height_cm: float
    medical_history: Optional[dict] = {}
    lifestyle: Optional[dict] = {}

@router.post("/")
def create_profile(req: ProfileRequest, db: Session = Depends(get_db),
                   current_user=Depends(get_current_user)):
    profile = HealthProfile(
        user_id=current_user,
        age=req.age,
        weight_kg=req.weight_kg,
        height_cm=req.height_cm,
        medical_history=req.medical_history,
        lifestyle=req.lifestyle,
    )
    db.add(profile)
    db.commit()
    return {"message": "profile saved"}

@router.get("/")
def get_profile(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    profile = db.query(HealthProfile).filter(HealthProfile.user_id == current_user).first()
    if not profile:
        return {"message": "no profile found"}
    return profile