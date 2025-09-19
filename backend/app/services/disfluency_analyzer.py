import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import Counter
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)

class DisfluencyAnalyzer:
    """
    Disfluency detection and analysis based on López-de Ipiña et al. (2013)
    Implements rule-based and ML-based detection of speech disfluencies
    """
    
    def __init__(self):
        # Initialize spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Disfluency patterns (rule-based)
        self.filled_pause_patterns = [
            r'\b(uh|um|er|ah|eh|mm|hmm|mhm)\b',
            r'\b(like|you know|I mean|sort of|kind of)\b',
            r'\b(well|so|actually|basically)\b'
        ]
        
        self.repetition_patterns = [
            r'\b(\w+)\s+\1\b',  # Word repetition: "the the"
            r'\b(\w+)\s+\1\s+\1\b',  # Triple repetition: "the the the"
        ]
        
        self.false_start_patterns = [
            r'\b\w+\s*-\s*\w+',  # "I was- I am"
            r'\b\w+\.\.\.\w+',   # "I was...I am"
        ]
        
        # Initialize ML-based disfluency detector (if available)
        self.ml_detector = None
        self._initialize_ml_detector()
    
    def _initialize_ml_detector(self):
        """Initialize ML-based disfluency detection model"""
        try:
            # This would use a pre-trained model for disfluency detection
            # For now, we'll use a placeholder - in production, you'd train/load a specific model
            logger.info("ML-based disfluency detector not implemented - using rule-based approach")
            self.ml_detector = None
        except Exception as e:
            logger.warning(f"Failed to initialize ML disfluency detector: {str(e)}")
            self.ml_detector = None
    
    def analyze_disfluencies(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive disfluency analysis of transcribed speech
        """
        try:
            transcript = transcription_result.get("transcript_text", "")
            word_timestamps = transcription_result.get("word_timestamps", [])
            
            if not transcript.strip():
                return self._empty_analysis()
            
            logger.info(f"Analyzing disfluencies in transcript: {len(transcript)} characters")
            
            # Rule-based analysis
            rule_based_results = self._rule_based_analysis(transcript, word_timestamps)
            
            # Pause analysis from timestamps
            pause_analysis = self._analyze_pauses(word_timestamps)
            
            # Combine results
            analysis_result = {
                **rule_based_results,
                **pause_analysis,
                "analysis_metadata": {
                    "transcript_length": len(transcript),
                    "word_count": len(transcript.split()),
                    "analysis_method": "rule_based_with_pauses"
                }
            }
            
            # Calculate disfluency rates
            analysis_result = self._calculate_disfluency_rates(analysis_result)
            
            logger.info("Disfluency analysis completed")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Disfluency analysis failed: {str(e)}")
            return self._empty_analysis()
    
    def _rule_based_analysis(self, transcript: str, word_timestamps: List[Dict]) -> Dict[str, Any]:
        """Rule-based disfluency detection"""
        try:
            results = {
                "filled_pauses_count": 0,
                "repetitions_count": 0,
                "false_starts_count": 0,
                "stutters_count": 0,
                "disfluency_events": []
            }
            
            # Normalize transcript
            normalized_text = transcript.lower().strip()
            
            # 1. Filled pauses detection
            filled_pauses = self._detect_filled_pauses(normalized_text, word_timestamps)
            results["filled_pauses_count"] = len(filled_pauses)
            results["disfluency_events"].extend(filled_pauses)
            
            # 2. Repetitions detection
            repetitions = self._detect_repetitions(normalized_text, word_timestamps)
            results["repetitions_count"] = len(repetitions)
            results["disfluency_events"].extend(repetitions)
            
            # 3. False starts detection
            false_starts = self._detect_false_starts(normalized_text, word_timestamps)
            results["false_starts_count"] = len(false_starts)
            results["disfluency_events"].extend(false_starts)
            
            # 4. Stutters detection (partial word repetitions)
            stutters = self._detect_stutters(normalized_text, word_timestamps)
            results["stutters_count"] = len(stutters)
            results["disfluency_events"].extend(stutters)
            
            # Sort events by timestamp
            results["disfluency_events"].sort(key=lambda x: x.get("start_time", 0))
            
            return results
            
        except Exception as e:
            logger.error(f"Rule-based analysis failed: {str(e)}")
            return {
                "filled_pauses_count": 0,
                "repetitions_count": 0,
                "false_starts_count": 0,
                "stutters_count": 0,
                "disfluency_events": []
            }
    
    def _detect_filled_pauses(self, text: str, word_timestamps: List[Dict]) -> List[Dict]:
        """Detect filled pauses (uh, um, er, etc.)"""
        filled_pauses = []
        
        for pattern in self.filled_pause_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group()
                
                # Find corresponding timestamp
                timestamp_info = self._find_timestamp_for_position(start_pos, end_pos, text, word_timestamps)
                
                filled_pauses.append({
                    "type": "filled_pause",
                    "text": matched_text,
                    "start_time": timestamp_info.get("start_time"),
                    "end_time": timestamp_info.get("end_time"),
                    "confidence": 0.8,  # Rule-based confidence
                    "pattern": pattern
                })
        
        return filled_pauses
    
    def _detect_repetitions(self, text: str, word_timestamps: List[Dict]) -> List[Dict]:
        """Detect word and phrase repetitions"""
        repetitions = []
        
        for pattern in self.repetition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group()
                
                timestamp_info = self._find_timestamp_for_position(start_pos, end_pos, text, word_timestamps)
                
                repetitions.append({
                    "type": "repetition",
                    "text": matched_text,
                    "start_time": timestamp_info.get("start_time"),
                    "end_time": timestamp_info.get("end_time"),
                    "confidence": 0.9,
                    "pattern": pattern
                })
        
        return repetitions
    
    def _detect_false_starts(self, text: str, word_timestamps: List[Dict]) -> List[Dict]:
        """Detect false starts and self-corrections"""
        false_starts = []
        
        for pattern in self.false_start_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group()
                
                timestamp_info = self._find_timestamp_for_position(start_pos, end_pos, text, word_timestamps)
                
                false_starts.append({
                    "type": "false_start",
                    "text": matched_text,
                    "start_time": timestamp_info.get("start_time"),
                    "end_time": timestamp_info.get("end_time"),
                    "confidence": 0.7,
                    "pattern": pattern
                })
        
        return false_starts
    
    def _detect_stutters(self, text: str, word_timestamps: List[Dict]) -> List[Dict]:
        """Detect stuttering (partial word repetitions)"""
        stutters = []
        
        # Pattern for partial word repetitions: "b-b-but", "th-th-that"
        stutter_pattern = r'\b(\w{1,3})-\1+\w*\b'
        matches = re.finditer(stutter_pattern, text, re.IGNORECASE)
        
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            matched_text = match.group()
            
            timestamp_info = self._find_timestamp_for_position(start_pos, end_pos, text, word_timestamps)
            
            stutters.append({
                "type": "stutter",
                "text": matched_text,
                "start_time": timestamp_info.get("start_time"),
                "end_time": timestamp_info.get("end_time"),
                "confidence": 0.8,
                "pattern": stutter_pattern
            })
        
        return stutters
    
    def _analyze_pauses(self, word_timestamps: List[Dict]) -> Dict[str, Any]:
        """Analyze silent pauses between words"""
        try:
            if not word_timestamps or len(word_timestamps) < 2:
                return {
                    "silent_pauses_count": 0,
                    "pause_durations": []
                }
            
            pause_durations = []
            silent_pauses = []
            
            for i in range(len(word_timestamps) - 1):
                current_word = word_timestamps[i]
                next_word = word_timestamps[i + 1]
                
                current_end = current_word.get("end")
                next_start = next_word.get("start")
                
                if current_end is not None and next_start is not None:
                    pause_duration = next_start - current_end
                    
                    # Consider pauses longer than 200ms as significant
                    if pause_duration > 0.2:
                        pause_durations.append(pause_duration)
                        
                        # Pauses longer than 500ms are considered silent pauses
                        if pause_duration > 0.5:
                            silent_pauses.append({
                                "type": "silent_pause",
                                "duration": pause_duration,
                                "start_time": current_end,
                                "end_time": next_start,
                                "before_word": current_word.get("word", ""),
                                "after_word": next_word.get("word", "")
                            })
            
            return {
                "silent_pauses_count": len(silent_pauses),
                "pause_durations": pause_durations,
                "silent_pause_events": silent_pauses
            }
            
        except Exception as e:
            logger.error(f"Pause analysis failed: {str(e)}")
            return {
                "silent_pauses_count": 0,
                "pause_durations": []
            }
    
    def _find_timestamp_for_position(self, start_pos: int, end_pos: int, text: str, 
                                   word_timestamps: List[Dict]) -> Dict[str, Optional[float]]:
        """Find timestamp information for text position"""
        try:
            if not word_timestamps:
                return {"start_time": None, "end_time": None}
            
            # Simple approach: find words that overlap with the text position
            # This is approximate since we don't have character-level alignment
            
            words_before = len(text[:start_pos].split())
            words_in_match = len(text[start_pos:end_pos].split())
            
            start_word_idx = max(0, words_before - 1)
            end_word_idx = min(len(word_timestamps) - 1, start_word_idx + words_in_match)
            
            start_time = word_timestamps[start_word_idx].get("start") if start_word_idx < len(word_timestamps) else None
            end_time = word_timestamps[end_word_idx].get("end") if end_word_idx < len(word_timestamps) else None
            
            return {"start_time": start_time, "end_time": end_time}
            
        except Exception as e:
            logger.error(f"Timestamp lookup failed: {str(e)}")
            return {"start_time": None, "end_time": None}
    
    def _calculate_disfluency_rates(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate disfluency rates per 100 words"""
        try:
            word_count = analysis_result.get("analysis_metadata", {}).get("word_count", 0)
            
            if word_count == 0:
                return analysis_result
            
            # Calculate rates per 100 words
            total_disfluencies = (
                analysis_result.get("filled_pauses_count", 0) +
                analysis_result.get("repetitions_count", 0) +
                analysis_result.get("false_starts_count", 0) +
                analysis_result.get("stutters_count", 0)
            )
            
            analysis_result["total_disfluency_rate"] = (total_disfluencies / word_count) * 100
            analysis_result["filled_pause_rate"] = (analysis_result.get("filled_pauses_count", 0) / word_count) * 100
            analysis_result["repetition_rate"] = (analysis_result.get("repetitions_count", 0) / word_count) * 100
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Rate calculation failed: {str(e)}")
            return analysis_result
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis result"""
        return {
            "filled_pauses_count": 0,
            "silent_pauses_count": 0,
            "repetitions_count": 0,
            "false_starts_count": 0,
            "stutters_count": 0,
            "total_disfluency_rate": 0.0,
            "filled_pause_rate": 0.0,
            "repetition_rate": 0.0,
            "disfluency_events": [],
            "pause_durations": [],
            "analysis_metadata": {
                "transcript_length": 0,
                "word_count": 0,
                "analysis_method": "empty"
            }
        }
    
    def generate_disfluency_timeline(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate an interactive timeline of disfluency events"""
        try:
            events = analysis_result.get("disfluency_events", [])
            silent_pauses = analysis_result.get("silent_pause_events", [])
            
            # Combine all events
            timeline_events = []
            
            # Add disfluency events
            for event in events:
                timeline_events.append({
                    "type": event.get("type"),
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                    "duration": (event.get("end_time", 0) - event.get("start_time", 0)) if event.get("start_time") and event.get("end_time") else None,
                    "text": event.get("text"),
                    "confidence": event.get("confidence"),
                    "category": "disfluency"
                })
            
            # Add silent pause events
            for pause in silent_pauses:
                timeline_events.append({
                    "type": "silent_pause",
                    "start_time": pause.get("start_time"),
                    "end_time": pause.get("end_time"),
                    "duration": pause.get("duration"),
                    "text": f"[{pause.get('duration', 0):.2f}s pause]",
                    "confidence": 1.0,
                    "category": "pause",
                    "context": f"{pause.get('before_word', '')} ... {pause.get('after_word', '')}"
                })
            
            # Sort by start time
            timeline_events.sort(key=lambda x: x.get("start_time", 0) or 0)
            
            return timeline_events
            
        except Exception as e:
            logger.error(f"Timeline generation failed: {str(e)}")
            return []
