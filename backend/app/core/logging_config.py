import logging
import sys
from app.core.config import settings

def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log")
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
