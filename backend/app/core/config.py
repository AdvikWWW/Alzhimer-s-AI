from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Alzheimer's Voice Biomarker Platform"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API
    API_V1_STR: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/alzheimers_voice"
    
    # Storage
    S3_BUCKET_NAME: str = "alzheimers-voice-recordings"
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    S3_REGION: str = "us-east-1"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML Models
    WHISPER_MODEL: str = "base"
    WHISPER_DEVICE: str = "cpu"  # or "cuda" if GPU available
    PYANNOTE_ACCESS_TOKEN: Optional[str] = None
    
    # Audio Processing
    SAMPLE_RATE: int = 16000
    MAX_AUDIO_SIZE_MB: int = 100
    SUPPORTED_AUDIO_FORMATS: List[str] = ["wav", "mp3", "m4a", "webm", "ogg"]
    
    # Feature Extraction
    FRAME_LENGTH: int = 2048
    HOP_LENGTH: int = 512
    N_MELS: int = 128
    N_MFCC: int = 13
    
    # Redis (for Celery)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
