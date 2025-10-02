from fastapi import APIRouter
from app.api.v1.endpoints import sessions, recordings, analysis, admin, analyze

api_router = APIRouter()

api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(recordings.router, prefix="/recordings", tags=["recordings"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(analyze.router, prefix="", tags=["analyze"])
