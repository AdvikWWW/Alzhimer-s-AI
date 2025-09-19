from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

Base = declarative_base()

class Participant(Base):
    __tablename__ = "participants"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    participant_id = Column(String, unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    native_language = Column(String, nullable=False)
    education_level = Column(String, nullable=False)
    hearing_impairment = Column(Boolean, default=False)
    speech_impairment = Column(Boolean, default=False)
    cognitive_status = Column(String, nullable=False)
    medications = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    consent_given = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    sessions = relationship("RecordingSession", back_populates="participant")

class RecordingSession(Base):
    __tablename__ = "recording_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, unique=True, nullable=False, index=True)
    participant_id = Column(String, ForeignKey("participants.id"), nullable=False)
    status = Column(String, default="in_progress")  # in_progress, completed, failed
    session_start_time = Column(DateTime(timezone=True), nullable=False)
    session_end_time = Column(DateTime(timezone=True), nullable=True)
    total_duration_seconds = Column(Integer, nullable=True)
    completed_tasks = Column(JSON, default=list)
    session_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    participant = relationship("Participant", back_populates="sessions")
    recordings = relationship("AudioRecording", back_populates="session")
    analysis_results = relationship("AnalysisResult", back_populates="session")

class AudioRecording(Base):
    __tablename__ = "audio_recordings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("recording_sessions.id"), nullable=False)
    task_id = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    channels = Column(Integer, nullable=False)
    format = Column(String, nullable=False)
    checksum = Column(String, nullable=False)
    upload_status = Column(String, default="uploaded")  # uploaded, processing, processed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("RecordingSession", back_populates="recordings")
    features = relationship("AudioFeatures", back_populates="recording")

class AudioFeatures(Base):
    __tablename__ = "audio_features"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id = Column(String, ForeignKey("audio_recordings.id"), nullable=False)
    
    # Acoustic Features
    pitch_mean = Column(Float, nullable=True)
    pitch_std = Column(Float, nullable=True)
    pitch_range = Column(Float, nullable=True)
    jitter_local = Column(Float, nullable=True)
    jitter_rap = Column(Float, nullable=True)
    shimmer_local = Column(Float, nullable=True)
    shimmer_apq3 = Column(Float, nullable=True)
    hnr_mean = Column(Float, nullable=True)
    hnr_std = Column(Float, nullable=True)
    
    # Voice Quality
    breathiness_score = Column(Float, nullable=True)
    hoarseness_score = Column(Float, nullable=True)
    vocal_tremor_score = Column(Float, nullable=True)
    
    # Speech Timing
    speech_rate_syllables_per_second = Column(Float, nullable=True)
    articulation_rate = Column(Float, nullable=True)
    pause_frequency = Column(Float, nullable=True)
    mean_pause_duration = Column(Float, nullable=True)
    
    # Spectral Features
    spectral_centroid_mean = Column(Float, nullable=True)
    spectral_bandwidth_mean = Column(Float, nullable=True)
    spectral_rolloff_mean = Column(Float, nullable=True)
    zero_crossing_rate_mean = Column(Float, nullable=True)
    mfcc_features = Column(JSON, nullable=True)  # Array of MFCC coefficients
    
    # Formant Features
    f1_mean = Column(Float, nullable=True)
    f2_mean = Column(Float, nullable=True)
    f3_mean = Column(Float, nullable=True)
    formant_dispersion = Column(Float, nullable=True)
    
    # Quality Flags
    signal_quality_score = Column(Float, nullable=True)
    noise_level = Column(Float, nullable=True)
    clipping_detected = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    recording = relationship("AudioRecording", back_populates="features")

class TranscriptionResult(Base):
    __tablename__ = "transcription_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id = Column(String, ForeignKey("audio_recordings.id"), nullable=False)
    engine = Column(String, nullable=False)  # whisperx, google, assemblyai
    transcript_text = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=True)
    word_timestamps = Column(JSON, nullable=True)  # Array of word-level timestamps
    speaker_labels = Column(JSON, nullable=True)  # Speaker diarization results
    language_detected = Column(String, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class DisfluencyAnalysis(Base):
    __tablename__ = "disfluency_analysis"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id = Column(String, ForeignKey("audio_recordings.id"), nullable=False)
    
    # Disfluency Counts
    filled_pauses_count = Column(Integer, default=0)
    silent_pauses_count = Column(Integer, default=0)
    repetitions_count = Column(Integer, default=0)
    false_starts_count = Column(Integer, default=0)
    stutters_count = Column(Integer, default=0)
    
    # Disfluency Rates (per 100 words)
    total_disfluency_rate = Column(Float, nullable=True)
    filled_pause_rate = Column(Float, nullable=True)
    repetition_rate = Column(Float, nullable=True)
    
    # Detailed Analysis
    disfluency_events = Column(JSON, nullable=True)  # Array of timestamped events
    pause_durations = Column(JSON, nullable=True)  # Array of pause durations
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class LexicalSemanticFeatures(Base):
    __tablename__ = "lexical_semantic_features"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id = Column(String, ForeignKey("audio_recordings.id"), nullable=False)
    
    # Lexical Diversity
    total_words = Column(Integer, nullable=True)
    unique_words = Column(Integer, nullable=True)
    type_token_ratio = Column(Float, nullable=True)
    moving_average_ttr = Column(Float, nullable=True)
    
    # Semantic Coherence
    semantic_coherence_score = Column(Float, nullable=True)
    topic_drift_score = Column(Float, nullable=True)
    idea_density = Column(Float, nullable=True)
    
    # Syntactic Complexity
    mean_sentence_length = Column(Float, nullable=True)
    syntactic_complexity_score = Column(Float, nullable=True)
    pos_tag_distribution = Column(JSON, nullable=True)
    
    # Word Frequency and Familiarity
    word_frequency_score = Column(Float, nullable=True)
    age_of_acquisition_score = Column(Float, nullable=True)
    
    # Semantic Embeddings
    sentence_embeddings = Column(JSON, nullable=True)  # Array of sentence embeddings
    semantic_similarity_matrix = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("recording_sessions.id"), nullable=False)
    
    # ML Model Predictions
    acoustic_model_score = Column(Float, nullable=True)
    lexical_model_score = Column(Float, nullable=True)
    combined_model_score = Column(Float, nullable=True)
    ensemble_score = Column(Float, nullable=True)
    
    # Risk Assessment
    risk_probability = Column(Float, nullable=True)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    uncertainty_score = Column(Float, nullable=True)
    
    # Quality Flags
    data_quality_score = Column(Float, nullable=True)
    requires_human_review = Column(Boolean, default=False)
    review_reasons = Column(JSON, nullable=True)
    
    # Model Metadata
    model_version = Column(String, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("RecordingSession", back_populates="analysis_results")

class QualityAssessment(Base):
    __tablename__ = "quality_assessments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("recording_sessions.id"), nullable=False)
    reviewer_id = Column(String, nullable=True)  # Human reviewer identifier
    
    # Review Status
    status = Column(String, default="pending")  # pending, in_review, completed
    review_start_time = Column(DateTime(timezone=True), nullable=True)
    review_end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Quality Scores
    audio_quality_score = Column(Float, nullable=True)
    transcription_quality_score = Column(Float, nullable=True)
    feature_extraction_quality_score = Column(Float, nullable=True)
    
    # Review Notes
    reviewer_notes = Column(Text, nullable=True)
    corrections_made = Column(JSON, nullable=True)
    final_recommendation = Column(String, nullable=True)  # accept, reject, reprocess
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
