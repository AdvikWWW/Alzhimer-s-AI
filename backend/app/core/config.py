from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str = "postgresql://alzheimer_user:alzheimer_pass@localhost:5432/alzheimer_voice_db"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # S3 Storage Configuration
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY_ID: str = "minioadmin"
    S3_SECRET_ACCESS_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "alzheimer-voice-data"
    S3_REGION: str = "us-east-1"
    
    # ML Model Configuration
    ML_MODELS_PATH: str = "./models"
    WHISPER_MODEL_SIZE: str = "base"
    ENABLE_GPU: bool = False
    
    # Audio Processing Configuration
    MAX_AUDIO_DURATION_MINUTES: int = 30
    AUDIO_SAMPLE_RATE: int = 16000
    VAD_AGGRESSIVENESS: int = 2
    
    # Application Configuration
    PROJECT_NAME: str = "Alzheimer Voice Biomarker Platform"
    DESCRIPTION: str = "Research platform for Alzheimer's disease detection using voice biomarkers"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: str = '["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"]'
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Development Configuration
    RELOAD: bool = True
    DEBUG: bool = True
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Additional ML Configuration
    PYANNOTE_ACCESS_TOKEN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
