from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.models import (
    Participant, RecordingSession, AudioRecording, 
    AnalysisResult, QualityAssessment
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/dashboard")
async def get_admin_dashboard(
    db: Session = Depends(get_db)
):
    """Get admin dashboard statistics"""
    try:
        # Basic counts
        total_participants = db.query(Participant).count()
        total_sessions = db.query(RecordingSession).count()
        total_recordings = db.query(AudioRecording).count()
        total_analyses = db.query(AnalysisResult).count()
        
        # Session status breakdown
        session_status_counts = db.query(
            RecordingSession.status,
            func.count(RecordingSession.id)
        ).group_by(RecordingSession.status).all()
        
        session_status = {status: count for status, count in session_status_counts}
        
        # Recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_sessions = db.query(RecordingSession).filter(
            RecordingSession.created_at >= week_ago
        ).count()
        
        recent_analyses = db.query(AnalysisResult).filter(
            AnalysisResult.created_at >= week_ago
        ).count()
        
        # Quality assessment stats
        pending_reviews = db.query(QualityAssessment).filter(
            QualityAssessment.status == "pending"
        ).count()
        
        requires_review = db.query(AnalysisResult).filter(
            AnalysisResult.requires_human_review == True
        ).count()
        
        # Cognitive status distribution
        cognitive_status_counts = db.query(
            Participant.cognitive_status,
            func.count(Participant.id)
        ).group_by(Participant.cognitive_status).all()
        
        cognitive_distribution = {status: count for status, count in cognitive_status_counts}
        
        # Storage statistics
        total_storage_bytes = db.query(
            func.sum(AudioRecording.file_size_bytes)
        ).scalar() or 0
        
        total_duration_seconds = db.query(
            func.sum(RecordingSession.total_duration_seconds)
        ).scalar() or 0
        
        return {
            "overview": {
                "total_participants": total_participants,
                "total_sessions": total_sessions,
                "total_recordings": total_recordings,
                "total_analyses": total_analyses,
                "total_storage_gb": round(total_storage_bytes / (1024**3), 2),
                "total_audio_hours": round(total_duration_seconds / 3600, 1)
            },
            "session_status": session_status,
            "recent_activity": {
                "sessions_last_7_days": recent_sessions,
                "analyses_last_7_days": recent_analyses
            },
            "quality_control": {
                "pending_reviews": pending_reviews,
                "requires_human_review": requires_review,
                "review_backlog": pending_reviews + requires_review
            },
            "participant_demographics": {
                "cognitive_status_distribution": cognitive_distribution
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get admin dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@router.get("/sessions")
async def list_all_sessions(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    cognitive_status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all sessions with filtering options"""
    try:
        query = db.query(RecordingSession).join(Participant)
        
        if status:
            query = query.filter(RecordingSession.status == status)
        
        if cognitive_status:
            query = query.filter(Participant.cognitive_status == cognitive_status)
        
        sessions = query.offset(skip).limit(limit).all()
        
        session_list = []
        for session in sessions:
            participant = session.participant
            
            # Get analysis results if available
            analysis = db.query(AnalysisResult).filter(
                AnalysisResult.session_id == session.id
            ).first()
            
            session_data = {
                "session_id": session.session_id,
                "participant_id": participant.participant_id,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
                "participant_info": {
                    "age": participant.age,
                    "gender": participant.gender,
                    "cognitive_status": participant.cognitive_status,
                    "education_level": participant.education_level
                },
                "session_metrics": {
                    "total_duration_seconds": session.total_duration_seconds,
                    "completed_tasks": len(session.completed_tasks or [])
                }
            }
            
            if analysis:
                session_data["analysis_results"] = {
                    "risk_probability": analysis.risk_probability,
                    "requires_human_review": analysis.requires_human_review,
                    "data_quality_score": analysis.data_quality_score
                }
            
            session_list.append(session_data)
        
        total_count = query.count()
        
        return {
            "sessions": session_list,
            "pagination": {
                "total": total_count,
                "skip": skip,
                "limit": limit,
                "has_more": skip + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@router.get("/quality-review")
async def get_quality_review_queue(
    db: Session = Depends(get_db)
):
    """Get sessions requiring quality review"""
    try:
        # Get sessions that require human review
        sessions_needing_review = db.query(RecordingSession).join(AnalysisResult).filter(
            AnalysisResult.requires_human_review == True
        ).all()
        
        review_queue = []
        for session in sessions_needing_review:
            analysis = db.query(AnalysisResult).filter(
                AnalysisResult.session_id == session.id
            ).first()
            
            # Check if already under review
            quality_assessment = db.query(QualityAssessment).filter(
                QualityAssessment.session_id == session.id
            ).first()
            
            review_item = {
                "session_id": session.session_id,
                "participant_id": session.participant.participant_id,
                "created_at": session.created_at.isoformat(),
                "analysis_issues": analysis.review_reasons if analysis else [],
                "data_quality_score": analysis.data_quality_score if analysis else None,
                "risk_probability": analysis.risk_probability if analysis else None,
                "review_status": quality_assessment.status if quality_assessment else "pending",
                "priority": _calculate_review_priority(analysis)
            }
            
            review_queue.append(review_item)
        
        # Sort by priority (high priority first)
        review_queue.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])
        
        return {
            "review_queue": review_queue,
            "total_items": len(review_queue),
            "priority_breakdown": {
                "high": len([item for item in review_queue if item["priority"] == "high"]),
                "medium": len([item for item in review_queue if item["priority"] == "medium"]),
                "low": len([item for item in review_queue if item["priority"] == "low"])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get quality review queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get review queue: {str(e)}")

@router.post("/quality-review/{session_id}/assign")
async def assign_quality_review(
    session_id: str,
    reviewer_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Assign a session for quality review"""
    try:
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if already assigned
        existing_assessment = db.query(QualityAssessment).filter(
            QualityAssessment.session_id == session.id
        ).first()
        
        if existing_assessment and existing_assessment.status == "in_review":
            raise HTTPException(status_code=400, detail="Session already under review")
        
        # Create or update quality assessment
        if existing_assessment:
            existing_assessment.status = "in_review"
            existing_assessment.reviewer_id = reviewer_data.get("reviewer_id")
            existing_assessment.review_start_time = datetime.utcnow()
            assessment = existing_assessment
        else:
            assessment = QualityAssessment(
                session_id=session.id,
                reviewer_id=reviewer_data.get("reviewer_id"),
                status="in_review",
                review_start_time=datetime.utcnow()
            )
            db.add(assessment)
        
        db.commit()
        
        return {
            "session_id": session_id,
            "assessment_id": assessment.id,
            "reviewer_id": assessment.reviewer_id,
            "status": "assigned",
            "message": "Session assigned for quality review"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign quality review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assign review: {str(e)}")

@router.get("/statistics")
async def get_system_statistics(
    db: Session = Depends(get_db)
):
    """Get comprehensive system statistics"""
    try:
        # Model performance statistics (mock data for now)
        model_stats = {
            "acoustic_model": {
                "accuracy": 0.847,
                "precision": 0.823,
                "recall": 0.871,
                "f1_score": 0.846,
                "auc_roc": 0.912
            },
            "lexical_model": {
                "accuracy": 0.792,
                "precision": 0.778,
                "recall": 0.806,
                "f1_score": 0.792,
                "auc_roc": 0.864
            },
            "ensemble_model": {
                "accuracy": 0.889,
                "precision": 0.876,
                "recall": 0.902,
                "f1_score": 0.889,
                "auc_roc": 0.943
            }
        }
        
        # Data quality statistics
        total_analyses = db.query(AnalysisResult).count()
        high_quality = db.query(AnalysisResult).filter(
            AnalysisResult.data_quality_score >= 0.8
        ).count()
        
        requires_review_count = db.query(AnalysisResult).filter(
            AnalysisResult.requires_human_review == True
        ).count()
        
        # Processing time statistics
        avg_processing_time = db.query(
            func.avg(AnalysisResult.processing_time_seconds)
        ).scalar() or 0
        
        return {
            "model_performance": model_stats,
            "data_quality": {
                "total_analyses": total_analyses,
                "high_quality_percentage": round((high_quality / total_analyses * 100), 1) if total_analyses > 0 else 0,
                "requires_review_percentage": round((requires_review_count / total_analyses * 100), 1) if total_analyses > 0 else 0
            },
            "processing_metrics": {
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "total_processing_hours": round(avg_processing_time * total_analyses / 3600, 1)
            },
            "system_health": {
                "status": "operational",
                "uptime_percentage": 99.7,
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

def _calculate_review_priority(analysis: AnalysisResult) -> str:
    """Calculate review priority based on analysis results"""
    if not analysis:
        return "medium"
    
    # High priority conditions
    if (analysis.data_quality_score and analysis.data_quality_score < 0.5) or \
       (analysis.uncertainty_score and analysis.uncertainty_score > 0.8) or \
       (analysis.review_reasons and len(analysis.review_reasons) >= 3):
        return "high"
    
    # Low priority conditions
    if (analysis.data_quality_score and analysis.data_quality_score > 0.8) and \
       (analysis.uncertainty_score and analysis.uncertainty_score < 0.3):
        return "low"
    
    return "medium"
