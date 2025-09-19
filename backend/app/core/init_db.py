import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.database import Base, engine
from app.core.config import settings
from app.models.models import *  # Import all models
from app.services.storage_service import storage_service

logger = logging.getLogger(__name__)

def create_database_if_not_exists():
    """Create database if it doesn't exist."""
    try:
        # Extract database name from URL
        db_url_parts = settings.DATABASE_URL.split('/')
        db_name = db_url_parts[-1]
        base_url = '/'.join(db_url_parts[:-1])
        
        # Connect to PostgreSQL server (not specific database)
        temp_engine = create_engine(f"{base_url}/postgres")
        
        with temp_engine.connect() as conn:
            # Set autocommit mode
            conn = conn.execution_options(autocommit=True)
            
            # Check if database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": db_name}
            )
            
            if not result.fetchone():
                # Create database
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info(f"Created database: {db_name}")
            else:
                logger.info(f"Database {db_name} already exists")
                
        temp_engine.dispose()
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise

def init_db():
    """Initialize database tables and setup."""
    try:
        logger.info("Initializing database...")
        
        # Create database if needed
        create_database_if_not_exists()
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Created database tables")
        
        # Initialize storage service
        logger.info("Initializing storage service...")
        storage_stats = storage_service.get_storage_stats()
        logger.info(f"Storage initialized: {storage_stats}")
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def reset_db():
    """Reset database by dropping and recreating all tables."""
    try:
        logger.warning("Resetting database - all data will be lost!")
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("Dropped all tables")
        
        # Recreate tables
        Base.metadata.create_all(bind=engine)
        logger.info("Recreated all tables")
        
        logger.info("Database reset completed")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_db()
    else:
        init_db()
