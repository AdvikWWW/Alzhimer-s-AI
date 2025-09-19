from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import hashlib
from datetime import datetime

from app.core.database import get_db
from app.models.models import AudioRecording, Session as SessionModel
from app.core.config import settings
from app.services.storage_service import storage_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/{recording_id}")
async def get_recording(
    recording_id: str,
    db: Session = Depends(get_db)
):
    """Get recording details"""
    try:
        recording = db.query(AudioRecording).filter(
            AudioRecording.id == recording_id
        ).first()
        
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        return {
            "recording_id": recording.id,
            "session_id": recording.session.session_id,
            "task_id": recording.task_id,
            "task_type": recording.task_type,
            "duration_seconds": recording.duration_seconds,
            "file_size_bytes": recording.file_size_bytes,
            "sample_rate": recording.sample_rate,
            "channels": recording.channels,
            "format": recording.format,
            "upload_status": recording.upload_status,
            "created_at": recording.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recording {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recording: {str(e)}")

@router.post("/upload")
async def upload_recording_file(
    session_id: str = Form(...),
    task_id: str = Form(...),
    task_type: str = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload an audio recording file"""
    try:
        # Validate session exists
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate file format
        if not audio_file.filename.lower().endswith(tuple(f".{fmt}" for fmt in settings.SUPPORTED_AUDIO_FORMATS)):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Supported formats: {settings.SUPPORTED_AUDIO_FORMATS}"
            )
        
        # Read and validate file size
        audio_content = await audio_file.read()
        file_size_mb = len(audio_content) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_AUDIO_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_AUDIO_SIZE_MB}MB"
            )
        
        # Calculate checksum
        checksum = hashlib.sha256(audio_content).hexdigest()
        
        # Check for duplicate uploads
        existing_recording = db.query(AudioRecording).filter(
            AudioRecording.session_id == session.id,
            AudioRecording.task_id == task_id,
            AudioRecording.checksum == checksum
        ).first()
        
        if existing_recording:
            return {
                "recording_id": existing_recording.id,
                "status": "duplicate",
                "message": "Recording already exists"
            }
        
        # Create storage directory
        storage_dir = Path("storage") / "recordings" / session_id
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_extension = Path(audio_file.filename).suffix
        filename = f"{task_id}_{checksum[:8]}{file_extension}"
        file_path = storage_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(audio_content)
        
        # Create database record
        recording = AudioRecording(
            session_id=session.id,
            task_id=task_id,
            task_type=task_type,
            file_path=str(file_path),
            file_size_bytes=len(audio_content),
            duration_seconds=0.0,  # Will be calculated during processing
            sample_rate=settings.SAMPLE_RATE,
            channels=1,
            format=file_extension.lstrip('.'),
            checksum=checksum,
            upload_status="uploaded"
        )
        
        db.add(recording)
        db.commit()
        
        logger.info(f"Uploaded recording {task_id} for session {session_id}")
        
        return {
            "recording_id": recording.id,
            "task_id": task_id,
            "file_size_bytes": len(audio_content),
            "checksum": checksum,
            "status": "uploaded",
            "message": "Recording uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload recording: {str(e)}")

@router.get("/session/{session_id}")
async def get_session_recordings(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get all recordings for a session"""
    try:
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        recordings = db.query(AudioRecording).filter(
            AudioRecording.session_id == session.id
        ).all()
        
        recording_list = []
        for recording in recordings:
            recording_list.append({
                "recording_id": recording.id,
                "task_id": recording.task_id,
                "task_type": recording.task_type,
                "duration_seconds": recording.duration_seconds,
                "file_size_bytes": recording.file_size_bytes,
                "upload_status": recording.upload_status,
                "created_at": recording.created_at.isoformat()
            })
        
        return {
            "session_id": session_id,
            "recordings": recording_list,
            "total_recordings": len(recording_list),
            "total_size_bytes": sum(r["file_size_bytes"] for r in recording_list),
            "total_duration_seconds": sum(r["duration_seconds"] for r in recording_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recordings for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session recordings: {str(e)}")

@router.delete("/{recording_id}")
async def delete_recording(
    recording_id: str,
    db: Session = Depends(get_db)
):
    """Delete a recording"""
    try:
        recording = db.query(AudioRecording).filter(
            AudioRecording.id == recording_id
        ).first()
        
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        # Delete file from storage
        file_path = Path(recording.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete database record
        db.delete(recording)
        db.commit()
        
        logger.info(f"Deleted recording {recording_id}")
        
        return {
            "recording_id": recording_id,
            "status": "deleted",
            "message": "Recording deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete recording {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete recording: {str(e)}")
