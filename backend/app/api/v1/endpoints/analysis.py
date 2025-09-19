from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime

from app.core.database import get_db
from app.models.models import RecordingSession, AnalysisResult, AudioRecording
from app.services.ml_service import MLService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/{session_id}/analyze")
async def analyze_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Trigger ML analysis for a completed session"""
    try:
        # Get session and validate
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status != "completed":
            raise HTTPException(status_code=400, detail="Session must be completed before analysis")
        
        # Check if analysis already exists
        existing_analysis = db.query(AnalysisResult).filter(
            AnalysisResult.session_id == session.id
        ).first()
        
        if existing_analysis:
            return {
                "session_id": session_id,
                "status": "already_analyzed",
                "analysis_id": existing_analysis.id,
                "message": "Analysis already exists for this session"
            }
        
        # Get ML service from app state
        ml_service: MLService = request.app.state.ml_service
        
        # Prepare session data for analysis
        recordings = db.query(AudioRecording).filter(
            AudioRecording.session_id == session.id
        ).all()
        
        session_data = {
            "session_id": session_id,
            "recordings": [(rec.task_id, b"dummy_audio_data") for rec in recordings],  # TODO: Load actual audio data
            "participant_data": {
                "age": session.participant.age,
                "gender": session.participant.gender,
                "cognitive_status": session.participant.cognitive_status
            }
        }
        
        # Run ML analysis
        analysis_results = await ml_service.analyze_recording_session(session_data)
        
        # Save analysis results to database
        ml_predictions = analysis_results.get("ml_predictions", {})
        risk_assessment = analysis_results.get("risk_assessment", {})
        quality_flags = analysis_results.get("quality_flags", {})
        
        analysis_record = AnalysisResult(
            session_id=session.id,
            acoustic_model_score=ml_predictions.get("acoustic_model", {}).get("probability"),
            lexical_model_score=ml_predictions.get("lexical_model", {}).get("probability"),
            combined_model_score=ml_predictions.get("combined_model", {}).get("probability"),
            ensemble_score=ml_predictions.get("ensemble_model", {}).get("probability"),
            risk_probability=risk_assessment.get("risk_probability"),
            confidence_interval_lower=risk_assessment.get("confidence_interval_lower"),
            confidence_interval_upper=risk_assessment.get("confidence_interval_upper"),
            uncertainty_score=risk_assessment.get("uncertainty_score"),
            data_quality_score=quality_flags.get("data_completeness"),
            requires_human_review=quality_flags.get("requires_human_review", False),
            review_reasons=quality_flags.get("issues", []),
            model_version="v1.0",
            feature_importance=json.dumps({}),  # TODO: Extract feature importance
            processing_time_seconds=0.0  # TODO: Track processing time
        )
        
        db.add(analysis_record)
        db.commit()
        
        logger.info(f"Analysis completed for session {session_id}")
        
        return {
            "session_id": session_id,
            "analysis_id": analysis_record.id,
            "status": "completed",
            "risk_assessment": risk_assessment,
            "quality_flags": quality_flags,
            "requires_human_review": analysis_record.requires_human_review,
            "message": "Analysis completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/{session_id}/results")
async def get_analysis_results(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get analysis results for a session"""
    try:
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        analysis = db.query(AnalysisResult).filter(
            AnalysisResult.session_id == session.id
        ).first()
        
        if not analysis:
            return {
                "session_id": session_id,
                "status": "not_analyzed",
                "message": "No analysis results found for this session"
            }
        
        # Get participant data
        participant = session.participant
        
        # Compile comprehensive results
        results = {
            "session_id": session_id,
            "analysis_id": analysis.id,
            "participant_info": {
                "participant_id": participant.participant_id,
                "age": participant.age,
                "gender": participant.gender,
                "cognitive_status": participant.cognitive_status,
                "education_level": participant.education_level
            },
            "session_info": {
                "session_start_time": session.session_start_time.isoformat(),
                "session_end_time": session.session_end_time.isoformat() if session.session_end_time else None,
                "total_duration_seconds": session.total_duration_seconds,
                "completed_tasks": session.completed_tasks
            },
            "ml_predictions": {
                "acoustic_model_score": analysis.acoustic_model_score,
                "lexical_model_score": analysis.lexical_model_score,
                "combined_model_score": analysis.combined_model_score,
                "ensemble_score": analysis.ensemble_score
            },
            "risk_assessment": {
                "risk_probability": analysis.risk_probability,
                "confidence_interval_lower": analysis.confidence_interval_lower,
                "confidence_interval_upper": analysis.confidence_interval_upper,
                "uncertainty_score": analysis.uncertainty_score,
                "risk_category": _get_risk_category(analysis.risk_probability)
            },
            "quality_assessment": {
                "data_quality_score": analysis.data_quality_score,
                "requires_human_review": analysis.requires_human_review,
                "review_reasons": analysis.review_reasons or [],
                "overall_quality": _get_overall_quality(analysis.data_quality_score)
            },
            "analysis_metadata": {
                "model_version": analysis.model_version,
                "processing_time_seconds": analysis.processing_time_seconds,
                "created_at": analysis.created_at.isoformat()
            },
            "research_disclaimer": {
                "message": "This analysis is for research purposes only and should not be used for clinical diagnosis.",
                "interpretation_required": "Results must be interpreted by qualified healthcare professionals.",
                "uncertainty_note": "All predictions include uncertainty measures and confidence intervals."
            }
        }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis results for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis results: {str(e)}")

@router.get("/{session_id}/timeline")
async def get_analysis_timeline(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get interactive timeline of disfluency events and biomarkers"""
    try:
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # TODO: Get detailed timeline data from stored analysis results
        # For now, return mock timeline data
        
        timeline_events = [
            {
                "type": "filled_pause",
                "start_time": 12.5,
                "end_time": 13.1,
                "duration": 0.6,
                "text": "um",
                "confidence": 0.9,
                "category": "disfluency",
                "task_id": "narrative"
            },
            {
                "type": "silent_pause",
                "start_time": 25.3,
                "end_time": 27.1,
                "duration": 1.8,
                "text": "[1.8s pause]",
                "confidence": 1.0,
                "category": "pause",
                "task_id": "narrative",
                "context": "was ... going"
            },
            {
                "type": "repetition",
                "start_time": 45.2,
                "end_time": 46.8,
                "duration": 1.6,
                "text": "I was I was",
                "confidence": 0.8,
                "category": "disfluency",
                "task_id": "picture"
            }
        ]
        
        # Group events by task
        timeline_by_task = {}
        for event in timeline_events:
            task_id = event.get("task_id", "unknown")
            if task_id not in timeline_by_task:
                timeline_by_task[task_id] = []
            timeline_by_task[task_id].append(event)
        
        return {
            "session_id": session_id,
            "timeline_events": timeline_events,
            "timeline_by_task": timeline_by_task,
            "summary": {
                "total_events": len(timeline_events),
                "disfluency_events": len([e for e in timeline_events if e["category"] == "disfluency"]),
                "pause_events": len([e for e in timeline_events if e["category"] == "pause"]),
                "total_duration": sum(e.get("duration", 0) for e in timeline_events)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get timeline for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")

@router.get("/{session_id}/biomarkers")
async def get_biomarker_details(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed biomarker analysis for a session"""
    try:
        session = db.query(RecordingSession).filter(
            RecordingSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # TODO: Get detailed biomarker data from stored features
        # For now, return mock biomarker data
        
        biomarkers = {
            "acoustic_features": {
                "pitch": {
                    "mean_f0": 185.4,
                    "std_f0": 42.1,
                    "range_f0": 156.3,
                    "percentile_25": 162.1,
                    "percentile_75": 208.7,
                    "interpretation": "Within normal range for age and gender"
                },
                "voice_quality": {
                    "jitter": 0.012,
                    "shimmer": 0.045,
                    "hnr": 18.6,
                    "breathiness_score": 0.23,
                    "interpretation": "Slight increase in breathiness, otherwise normal"
                },
                "timing": {
                    "speech_rate": 4.2,
                    "articulation_rate": 5.1,
                    "pause_frequency": 8.3,
                    "mean_pause_duration": 0.85,
                    "interpretation": "Slightly reduced speech rate with increased pausing"
                }
            },
            "lexical_semantic_features": {
                "lexical_diversity": {
                    "type_token_ratio": 0.68,
                    "moving_average_ttr": 0.71,
                    "vocabulary_size": 156,
                    "interpretation": "Good lexical diversity for narrative tasks"
                },
                "semantic_coherence": {
                    "coherence_score": 0.74,
                    "topic_drift": 0.12,
                    "idea_density": 0.58,
                    "interpretation": "Maintained semantic coherence throughout tasks"
                },
                "syntactic_complexity": {
                    "mean_sentence_length": 8.4,
                    "complexity_score": 0.45,
                    "clause_density": 1.2,
                    "interpretation": "Moderate syntactic complexity, appropriate for tasks"
                }
            },
            "disfluency_patterns": {
                "filled_pauses": {
                    "count": 12,
                    "rate_per_100_words": 4.8,
                    "most_common": ["um", "uh", "er"],
                    "interpretation": "Slightly elevated filled pause rate"
                },
                "repetitions": {
                    "count": 3,
                    "rate_per_100_words": 1.2,
                    "types": ["word", "phrase"],
                    "interpretation": "Normal repetition rate"
                },
                "silent_pauses": {
                    "count": 18,
                    "mean_duration": 0.85,
                    "long_pauses": 4,
                    "interpretation": "Increased silent pausing, may indicate word-finding difficulty"
                }
            }
        }
        
        return {
            "session_id": session_id,
            "biomarkers": biomarkers,
            "clinical_interpretation": {
                "summary": "Mixed pattern with some indicators of mild cognitive changes",
                "key_findings": [
                    "Increased silent pausing frequency",
                    "Slightly elevated filled pause rate",
                    "Maintained semantic coherence",
                    "Normal lexical diversity"
                ],
                "recommendations": [
                    "Follow-up assessment recommended",
                    "Monitor changes over time",
                    "Consider comprehensive neuropsychological evaluation"
                ]
            },
            "research_context": {
                "reference_studies": [
                    "López-de Ipiña et al. (2013) - Disfluency patterns",
                    "Saeedi et al. (2024) - Voice quality markers",
                    "Favaro et al. (2023) - Semantic coherence",
                    "Yang et al. (2022) - Speech timing measures"
                ],
                "normative_data": "Compared against DementiaBank and ADReSS datasets"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get biomarkers for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get biomarkers: {str(e)}")

def _get_risk_category(risk_probability: Optional[float]) -> str:
    """Determine risk category from probability"""
    if risk_probability is None:
        return "unknown"
    elif risk_probability < 0.3:
        return "low"
    elif risk_probability < 0.7:
        return "moderate"
    else:
        return "high"

def _get_overall_quality(data_quality_score: Optional[float]) -> str:
    """Determine overall quality from score"""
    if data_quality_score is None:
        return "unknown"
    elif data_quality_score >= 0.8:
        return "excellent"
    elif data_quality_score >= 0.6:
        return "good"
    elif data_quality_score >= 0.4:
        return "fair"
    else:
        return "poor"
