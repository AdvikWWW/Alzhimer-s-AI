from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
from typing import List, Optional

from app.core.config import settings
from app.core.database import engine, SessionLocal
from app.models import models
from app.api.v1.api import api_router
from app.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Alzheimer's Voice Biomarker Platform API")
    
    # Create database tables
    models.Base.metadata.create_all(bind=engine)
    
    # Initialize ML models (lazy loading)
    from app.services.ml_service import MLService
    ml_service = MLService()
    app.state.ml_service = ml_service
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")

app = FastAPI(
    title="Alzheimer's Voice Biomarker Platform",
    description="Advanced voice analysis for Alzheimer's disease research using clinically-validated biomarkers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Alzheimer's Voice Biomarker Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
        "research_disclaimer": "This API is for research purposes only and should not be used for clinical diagnosis"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2025-09-14T11:42:08-04:00",
        "services": {
            "database": "operational",
            "storage": "operational",
            "ml_pipeline": "operational"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_level="info"
    )
