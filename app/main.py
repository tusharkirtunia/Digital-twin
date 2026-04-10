from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, auth, profile, twin
from app.core.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Health Digital Twin API", version="1.0")

# CORS — allows frontend/mobile to call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route groups
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(profile.router)
app.include_router(twin.router)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.get("/")
def root():
    return {"status": "Health Digital Twin API is running"}