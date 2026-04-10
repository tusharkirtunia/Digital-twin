from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.auth import router as auth_router
from app.api.profile import router as profile_router

app = FastAPI()

app.include_router(auth_router, prefix="/auth")
app.include_router(health_router, prefix="/health")
app.include_router(profile_router, prefix="/profile")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)