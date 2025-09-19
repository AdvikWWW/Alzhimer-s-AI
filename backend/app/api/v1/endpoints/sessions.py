from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
import uuid

from app.core.database import get_db
from app.models.models import Participant, RecordingSession, AudioRecording
from app.services.ml_service import MLService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/create")
async def create_session(
    participant_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Create a new recording session with participant data"""
    try:
        # Create participant record
        participant = Participant(
            participant_id=participant_data["participantId"],
            age=participant_data["age"],
            gender=participant_data["gender"],
            native_language=participant_data["nativeLanguage"],
            education_level=participant_data["educationLevel"],
            hearing_impairment=participant_data.get("hearingImpairment", False),
            speech_impairment=participant_data.get("speechImpairment", False),
            cognitive_status=participant_data["cognitiveStatus"],
            medications=participant_data.get("medications"),
            notes=participant_data.get("notes"),
            consent_given=participant_data["consentGiven"]
        )
        
        db.add(participant)
        db.flush()  # Get the participant ID
        
        # Create recording session
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        session = RecordingSession(
            session_id=session_id,
            participant_id=participant.id,
            session_start_time=datetime.utcnow(),
            status="in_progress"
        )
        
        db.add(session)
        db.commit()
        
        logger.info(f"Created session {session_id} for participant {participant.participant_id}")
        
        return {
            "session_id": session_id,
            "participant_id": participant.participant_id,
            "status": "created",
            "message": "Session created successfully"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.get("/{session_id}")
async def get_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get session details"""
    try:
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        participant = db.query(Participant).filter(
            Participant.id == session.participant_id
        ).first()
        
        recordings = db.query(AudioRecording).filter(
            AudioRecording.session_id == session.id
        ).all()
        
        return {
            "session_id": session.session_id,
            "status": session.status,
            "participant": {
                "participant_id": participant.participant_id,
                "age": participant.age,
                "gender": participant.gender,
                "cognitive_status": participant.cognitive_status
            },
            "session_start_time": session.session_start_time.isoformat(),
            "session_end_time": session.session_end_time.isoformat() if session.session_end_time else None,
            "completed_tasks": session.completed_tasks,
            "recordings": [
                {
                    "task_id": rec.task_id,
                    "task_type": rec.task_type,
                    "duration_seconds": rec.duration_seconds,
                    "upload_status": rec.upload_status
                }
                for rec in recordings
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@router.post("/{session_id}/upload")
async def upload_recording(
    session_id: str,
    task_id: str = Form(...),
    task_type: str = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload an audio recording for a session"""
    try:
        # Validate session exists
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Read audio file
        audio_content = await audio_file.read()
        
        # TODO: Save to S3 or file storage
        # For now, we'll store file info in database
        file_path = f"recordings/{session_id}/{task_id}.wav"
        
        # Create audio recording record
        recording = AudioRecording(
            session_id=session.id,
            task_id=task_id,
            task_type=task_type,
            file_path=file_path,
            file_size_bytes=len(audio_content),
            duration_seconds=0.0,  # Will be calculated during processing
            sample_rate=16000,  # Default
            channels=1,  # Default
            format="wav",
            checksum="",  # TODO: Calculate checksum
            upload_status="uploaded"
        )
        
        db.add(recording)
        db.commit()
        
        logger.info(f"Uploaded recording {task_id} for session {session_id}")
        
        return {
            "recording_id": recording.id,
            "task_id": task_id,
            "status": "uploaded",
            "message": "Recording uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload recording: {str(e)}")

@router.post("/{session_id}/complete")
async def complete_session(
    session_id: str,
    session_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Complete a recording session and trigger analysis"""
    try:
        # Update session status
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session.status = "completed"
        session.session_end_time = datetime.utcnow()
        session.completed_tasks = session_data.get("completedTasks", [])
        session.total_duration_seconds = session_data.get("totalDuration", 0)
        
        db.commit()
        
        # Trigger ML analysis (async)
        # TODO: Use Celery for background processing
        # For now, we'll return success and process later
        
        logger.info(f"Completed session {session_id}")
        
        return {
            "session_id": session_id,
            "status": "completed",
            "message": "Session completed successfully",
            "analysis_status": "queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to complete session: {str(e)}")

@router.get("/{session_id}/status")
async def get_session_status(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get current session status and progress"""
    try:
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        recordings = db.query(AudioRecording).filter(
            AudioRecording.session_id == session.id
        ).all()
        
        return {
            "session_id": session.session_id,
            "status": session.status,
            "completed_tasks": session.completed_tasks or [],
            "total_recordings": len(recordings),
            "processing_status": {
                "uploaded": len([r for r in recordings if r.upload_status == "uploaded"]),
                "processing": len([r for r in recordings if r.upload_status == "processing"]),
                "processed": len([r for r in recordings if r.upload_status == "processed"]),
                "failed": len([r for r in recordings if r.upload_status == "failed"])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@router.get("/")
async def list_sessions(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List recording sessions with optional filtering"""
    try:
        query = db.query(RecordingSession)
        
        if status:
            query = query.filter(RecordingSession.status == status)
        
        sessions = query.offset(skip).limit(limit).all()
        
        session_list = []
        for session in sessions:
            participant = db.query(Participant).filter(
                Participant.id == session.participant_id
            ).first()
            
            session_list.append({
                "session_id": session.session_id,
                "status": session.status,
                "participant_id": participant.participant_id if participant else None,
                "session_start_time": session.session_start_time.isoformat(),
                "session_end_time": session.session_end_time.isoformat() if session.session_end_time else None,
                "total_duration_seconds": session.total_duration_seconds
            })
        
        return {
            "sessions": session_list,
            "total": len(session_list),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")
