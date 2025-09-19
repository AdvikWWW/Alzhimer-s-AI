import whisperx
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os
from pathlib import Path
import json

from app.core.config import settings

logger = logging.getLogger(__name__)

class ASRService:
    """
    Automatic Speech Recognition service using WhisperX with fallback options
    Implements multi-engine ASR with forced alignment for precise timing
    """
    
    def __init__(self):
        self.device = settings.WHISPER_DEVICE
        self.model_size = settings.WHISPER_MODEL
        self.model = None
        self.align_model = None
        self.diarize_model = None
        
        # Initialize models lazily
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize WhisperX models"""
        try:
            logger.info(f"Initializing WhisperX model: {self.model_size} on {self.device}")
            
            # Load WhisperX model
            self.model = whisperx.load_model(
                self.model_size, 
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            
            logger.info("WhisperX model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WhisperX model: {str(e)}")
            self.model = None
    
    def _initialize_alignment_model(self, language_code: str = "en"):
        """Initialize alignment model for the detected language"""
        try:
            if self.align_model is None:
                logger.info(f"Loading alignment model for language: {language_code}")
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code, 
                    device=self.device
                )
                self.align_model = (model_a, metadata)
                logger.info("Alignment model loaded successfully")
            
            return self.align_model
            
        except Exception as e:
            logger.error(f"Failed to load alignment model: {str(e)}")
            return None
    
    def _initialize_diarization_model(self):
        """Initialize speaker diarization model"""
        try:
            if self.diarize_model is None and settings.PYANNOTE_ACCESS_TOKEN:
                logger.info("Loading diarization model")
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=settings.PYANNOTE_ACCESS_TOKEN,
                    device=self.device
                )
                logger.info("Diarization model loaded successfully")
            
            return self.diarize_model
            
        except Exception as e:
            logger.error(f"Failed to load diarization model: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file using WhisperX with forced alignment
        """
        try:
            if self.model is None:
                raise RuntimeError("WhisperX model not initialized")
            
            logger.info(f"Starting transcription for: {audio_file_path}")
            
            # Load audio
            audio = whisperx.load_audio(audio_file_path)
            
            # Step 1: Transcribe with Whisper
            result = self.model.transcribe(audio, batch_size=16)
            
            # Extract language and segments
            language_code = result.get("language", "en")
            segments = result.get("segments", [])
            
            logger.info(f"Initial transcription completed. Language: {language_code}, Segments: {len(segments)}")
            
            # Step 2: Forced alignment for word-level timestamps
            align_model = self._initialize_alignment_model(language_code)
            
            if align_model:
                model_a, metadata = align_model
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=False
                )
                logger.info("Forced alignment completed")
            
            # Step 3: Speaker diarization (optional)
            diarize_model = self._initialize_diarization_model()
            
            if diarize_model:
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("Speaker diarization completed")
            
            # Process results
            transcription_result = self._process_whisperx_result(result, language_code)
            
            logger.info("Transcription processing completed successfully")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def _process_whisperx_result(self, result: Dict[str, Any], language_code: str) -> Dict[str, Any]:
        """Process WhisperX result into standardized format"""
        try:
            segments = result.get("segments", [])
            
            # Extract full transcript
            full_transcript = " ".join([segment.get("text", "").strip() for segment in segments])
            
            # Extract word-level timestamps
            word_timestamps = []
            speaker_labels = []
            
            for segment in segments:
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", 0)
                segment_text = segment.get("text", "").strip()
                segment_speaker = segment.get("speaker", "SPEAKER_00")
                
                # Word-level data
                words = segment.get("words", [])
                
                if words:
                    # Use word-level timestamps if available
                    for word_info in words:
                        word_timestamps.append({
                            "word": word_info.get("word", ""),
                            "start": word_info.get("start", segment_start),
                            "end": word_info.get("end", segment_end),
                            "confidence": word_info.get("score", 0.0),
                            "speaker": word_info.get("speaker", segment_speaker)
                        })
                else:
                    # Fallback to segment-level timestamps
                    words_in_segment = segment_text.split()
                    segment_duration = segment_end - segment_start
                    word_duration = segment_duration / len(words_in_segment) if words_in_segment else 0
                    
                    for i, word in enumerate(words_in_segment):
                        word_start = segment_start + (i * word_duration)
                        word_end = word_start + word_duration
                        
                        word_timestamps.append({
                            "word": word,
                            "start": word_start,
                            "end": word_end,
                            "confidence": segment.get("avg_logprob", 0.0),
                            "speaker": segment_speaker
                        })
                
                # Speaker information
                speaker_labels.append({
                    "start": segment_start,
                    "end": segment_end,
                    "speaker": segment_speaker,
                    "text": segment_text
                })
            
            # Calculate overall confidence
            confidences = [word.get("confidence", 0) for word in word_timestamps if word.get("confidence")]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "engine": "whisperx",
                "transcript_text": full_transcript,
                "confidence_score": overall_confidence,
                "language_detected": language_code,
                "word_timestamps": word_timestamps,
                "speaker_labels": speaker_labels,
                "segments": segments,
                "processing_metadata": {
                    "model_size": self.model_size,
                    "device": self.device,
                    "num_segments": len(segments),
                    "num_words": len(word_timestamps),
                    "num_speakers": len(set(label["speaker"] for label in speaker_labels))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process WhisperX result: {str(e)}")
            raise
    
    def transcribe_with_fallback(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe with fallback to alternative engines if WhisperX fails
        """
        try:
            # Primary: WhisperX
            return self.transcribe_audio(audio_file_path)
            
        except Exception as e:
            logger.warning(f"WhisperX transcription failed: {str(e)}")
            
            # Fallback options could include:
            # - Google Speech-to-Text API
            # - AssemblyAI API
            # - Azure Speech Services
            # For now, we'll raise the original error
            
            logger.error("No fallback transcription engines available")
            raise
    
    def validate_transcription_quality(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate transcription quality and flag potential issues
        """
        try:
            quality_flags = {
                "overall_quality": "good",
                "issues": [],
                "confidence_score": transcription_result.get("confidence_score", 0.0),
                "word_count": len(transcription_result.get("word_timestamps", [])),
                "speaker_consistency": True
            }
            
            # Check confidence score
            confidence = transcription_result.get("confidence_score", 0.0)
            if confidence < 0.3:
                quality_flags["overall_quality"] = "poor"
                quality_flags["issues"].append("Very low confidence score")
            elif confidence < 0.6:
                quality_flags["overall_quality"] = "fair"
                quality_flags["issues"].append("Low confidence score")
            
            # Check transcript length
            transcript = transcription_result.get("transcript_text", "")
            if len(transcript.strip()) < 10:
                quality_flags["overall_quality"] = "poor"
                quality_flags["issues"].append("Transcript too short")
            
            # Check for repeated words (potential transcription errors)
            words = transcript.lower().split()
            if len(words) > 0:
                repeated_ratio = len(words) - len(set(words))
                if repeated_ratio / len(words) > 0.3:
                    quality_flags["issues"].append("High repetition rate")
            
            # Check word timestamp coverage
            word_timestamps = transcription_result.get("word_timestamps", [])
            words_with_timestamps = sum(1 for w in word_timestamps if w.get("start") is not None)
            if word_timestamps and words_with_timestamps / len(word_timestamps) < 0.8:
                quality_flags["issues"].append("Poor timestamp coverage")
            
            # Determine if human review is needed
            quality_flags["requires_human_review"] = (
                quality_flags["overall_quality"] == "poor" or
                len(quality_flags["issues"]) >= 2
            )
            
            return quality_flags
            
        except Exception as e:
            logger.error(f"Transcription quality validation failed: {str(e)}")
            return {
                "overall_quality": "unknown",
                "issues": ["Quality validation failed"],
                "requires_human_review": True
            }
    
    def extract_speech_segments(self, transcription_result: Dict[str, Any]) -> List[Tuple[float, float]]:
        """
        Extract speech segments from transcription results
        """
        try:
            segments = []
            word_timestamps = transcription_result.get("word_timestamps", [])
            
            if not word_timestamps:
                return segments
            
            # Group consecutive words into segments
            current_start = None
            current_end = None
            gap_threshold = 0.5  # 500ms gap threshold
            
            for word in word_timestamps:
                word_start = word.get("start")
                word_end = word.get("end")
                
                if word_start is None or word_end is None:
                    continue
                
                if current_start is None:
                    current_start = word_start
                    current_end = word_end
                elif word_start - current_end <= gap_threshold:
                    # Extend current segment
                    current_end = word_end
                else:
                    # Start new segment
                    segments.append((current_start, current_end))
                    current_start = word_start
                    current_end = word_end
            
            # Add final segment
            if current_start is not None and current_end is not None:
                segments.append((current_start, current_end))
            
            logger.info(f"Extracted {len(segments)} speech segments from transcription")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to extract speech segments: {str(e)}")
            return []
