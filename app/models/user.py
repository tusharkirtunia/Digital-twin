# app/models/user.py
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
import uuid

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime)

class HealthProfile(Base):
    __tablename__ = "health_profiles"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    age = Column(Float)
    weight_kg = Column(Float)
    height_cm = Column(Float)
    medical_history = Column(JSON)   # e.g. {"conditions": ["hypertension"]}
    lifestyle = Column(JSON)          # e.g. {"smoker": false, "diet": "balanced"}

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    prediction_type = Column(String)  # e.g. "heart_disease", "stress"
    risk_score = Column(Float)
    explanation = Column(JSON)        # SHAP values
    created_at = Column(DateTime)