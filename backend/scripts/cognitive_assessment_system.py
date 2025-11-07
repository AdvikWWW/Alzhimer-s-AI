#!/usr/bin/env python3
"""
Advanced Cognitive Assessment System for Alzheimer's Detection
Real-time listening, task-based evaluation, and cognitive scoring
"""

import numpy as np
import torch
import librosa
import sounddevice as sd
from typing import Dict, List, Any, Optional, Tuple
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.audio_processor import AudioProcessor
from app.services.asr_service import ASRService
from app.services.disfluency_analyzer import DisfluencyAnalyzer
from app.services.lexical_semantic_analyzer import LexicalSemanticAnalyzer

# Import enhanced analyzer
from enhanced_word_level_analyzer import WordLevelAnalyzer, IntelligentAlzheimerScorer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CognitiveTask(Enum):
    """Types of cognitive assessment tasks"""
    PICTURE_DESCRIPTION = "picture_description"
    STORY_RECALL = "story_recall"
    VERBAL_FLUENCY = "verbal_fluency"
    SERIAL_SUBTRACTION = "serial_subtraction"
    FREE_SPEECH = "free_speech"


@dataclass
class TaskPrompt:
    """Task prompt configuration"""
    task_type: CognitiveTask
    prompt_text: str
    expected_keywords: List[str]
    duration_seconds: int
    scoring_weights: Dict[str, float]


class CognitiveAssessmentSystem:
    """
    Advanced cognitive assessment system with real-time analysis
    """
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.asr_service = ASRService()
        self.disfluency_analyzer = DisfluencyAnalyzer()
        self.lexical_analyzer = LexicalSemanticAnalyzer()
        self.word_analyzer = WordLevelAnalyzer(use_gpu=torch.cuda.is_available())
        self.scorer = IntelligentAlzheimerScorer()
        
        # Real-time audio processing
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # Process audio in 500ms chunks
        
        # Task configurations
        self.tasks = self._initialize_tasks()
        self.current_task = None
        self.task_start_time = None
        
        # Results storage
        self.session_results = []
        self.real_time_metrics = {
            'speech_rate': [],
            'pause_density': [],
            'coherence': [],
            'fluency': []
        }
    
    def _initialize_tasks(self) -> Dict[CognitiveTask, TaskPrompt]:
        """Initialize cognitive assessment tasks"""
        return {
            CognitiveTask.PICTURE_DESCRIPTION: TaskPrompt(
                task_type=CognitiveTask.PICTURE_DESCRIPTION,
                prompt_text="Please describe what you see happening in this picture. Include details about the people, objects, and actions.",
                expected_keywords=["person", "people", "action", "object", "scene", "happening", "doing"],
                duration_seconds=60,
                scoring_weights={
                    'fluency': 0.25,
                    'coherence': 0.25,
                    'relevance': 0.25,
                    'detail': 0.25
                }
            ),
            CognitiveTask.STORY_RECALL: TaskPrompt(
                task_type=CognitiveTask.STORY_RECALL,
                prompt_text="I'll tell you a short story. Please listen carefully and then repeat it back to me:\n'Sarah went to the grocery store. She bought apples, bread, and milk. On her way home, she met her neighbor Tom. They talked about the weather.'",
                expected_keywords=["Sarah", "grocery", "store", "apples", "bread", "milk", "neighbor", "Tom", "weather"],
                duration_seconds=45,
                scoring_weights={
                    'accuracy': 0.35,
                    'completeness': 0.35,
                    'fluency': 0.15,
                    'coherence': 0.15
                }
            ),
            CognitiveTask.VERBAL_FLUENCY: TaskPrompt(
                task_type=CognitiveTask.VERBAL_FLUENCY,
                prompt_text="Name as many animals as you can in the next 30 seconds. Begin now.",
                expected_keywords=[],  # Will be populated dynamically
                duration_seconds=30,
                scoring_weights={
                    'quantity': 0.40,
                    'uniqueness': 0.30,
                    'fluency': 0.15,
                    'clustering': 0.15
                }
            ),
            CognitiveTask.SERIAL_SUBTRACTION: TaskPrompt(
                task_type=CognitiveTask.SERIAL_SUBTRACTION,
                prompt_text="Start from 100 and subtract 7 each time. Continue as far as you can.",
                expected_keywords=["100", "93", "86", "79", "72", "65", "58", "51", "44", "37", "30", "23", "16", "9", "2"],
                duration_seconds=60,
                scoring_weights={
                    'accuracy': 0.40,
                    'progression': 0.30,
                    'fluency': 0.15,
                    'self_correction': 0.15
                }
            )
        }
    
    def start_task(self, task_type: CognitiveTask) -> Dict[str, Any]:
        """Start a cognitive assessment task"""
        self.current_task = self.tasks[task_type]
        self.task_start_time = time.time()
        self.session_results = []
        self.real_time_metrics = {
            'speech_rate': [],
            'pause_density': [],
            'coherence': [],
            'fluency': []
        }
        
        logger.info(f"Starting task: {task_type.value}")
        
        return {
            'task': task_type.value,
            'prompt': self.current_task.prompt_text,
            'duration': self.current_task.duration_seconds,
            'status': 'started'
        }
    
    def start_real_time_recording(self):
        """Start real-time audio recording"""
        self.is_recording = True
        self.audio_queue = queue.Queue()
        
        def audio_callback(indata, frames, time_info, status):
            """Callback for audio recording"""
            if status:
                logger.warning(f"Audio recording status: {status}")
            if self.is_recording:
                self.audio_queue.put(indata.copy())
        
        # Start recording
        self.stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * self.chunk_duration)
        )
        self.stream.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_chunks)
        self.processing_thread.start()
        
        logger.info("Real-time recording started")
    
    def stop_real_time_recording(self) -> Dict[str, Any]:
        """Stop real-time recording and return results"""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
        
        # Calculate final scores
        final_results = self._calculate_task_score()
        
        logger.info("Real-time recording stopped")
        return final_results
    
    def _process_audio_chunks(self):
        """Process audio chunks in real-time"""
        audio_buffer = []
        last_process_time = time.time()
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)
                
                # Process every 2 seconds
                if time.time() - last_process_time >= 2.0:
                    # Combine chunks
                    audio_data = np.concatenate(audio_buffer)
                    audio_buffer = []
                    last_process_time = time.time()
                    
                    # Process audio
                    self._analyze_audio_segment(audio_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio chunk: {str(e)}")
    
    def _analyze_audio_segment(self, audio_data: np.ndarray):
        """Analyze an audio segment"""
        try:
            # Save temporary audio file
            temp_path = Path(f"/tmp/segment_{time.time()}.wav")
            librosa.output.write_wav(str(temp_path), audio_data, self.sample_rate)
            
            # Extract features
            acoustic_features = self.audio_processor.process_audio_file(str(temp_path))
            
            # Transcribe
            transcription = self.asr_service.transcribe_audio(str(temp_path))
            
            # Analyze disfluencies
            disfluency_features = self.disfluency_analyzer.analyze_disfluencies(transcription)
            
            # Analyze lexical features
            lexical_features = self.lexical_analyzer.analyze_lexical_semantic_features(transcription)
            
            # Word-level analysis
            word_timestamps = transcription.get('word_timestamps', [])
            if word_timestamps:
                word_analysis = self.word_analyzer.analyze_audio_word_by_word(
                    str(temp_path), word_timestamps
                )
            else:
                word_analysis = {'aggregated_features': {}}
            
            # Calculate segment metrics
            segment_metrics = self._calculate_segment_metrics(
                acoustic_features, 
                disfluency_features, 
                lexical_features,
                word_analysis['aggregated_features']
            )
            
            # Update real-time metrics
            self._update_real_time_metrics(segment_metrics)
            
            # Store results
            self.session_results.append({
                'timestamp': time.time() - self.task_start_time,
                'transcription': transcription.get('transcript_text', ''),
                'metrics': segment_metrics
            })
            
            # Clean up
            temp_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Error analyzing audio segment: {str(e)}")
    
    def _calculate_segment_metrics(self, acoustic: Dict, disfluency: Dict, 
                                  lexical: Dict, word_level: Dict) -> Dict[str, float]:
        """Calculate metrics for an audio segment"""
        
        # Speech fluency score (0-1)
        fluency = 1.0
        fluency -= disfluency.get('filled_pause_rate', 0) * 2  # Penalize filled pauses
        fluency -= disfluency.get('repetition_rate', 0) * 3    # Penalize repetitions
        fluency -= acoustic.get('pause_time_ratio', 0) * 1.5   # Penalize excessive pausing
        fluency = max(0, min(1, fluency))
        
        # Coherence score (0-1)
        coherence = lexical.get('semantic_coherence_score', 0.5)
        
        # Speech rate (words per minute)
        speech_rate = acoustic.get('speech_rate_wpm', 0)
        
        # Pause density
        pause_density = acoustic.get('pause_time_ratio', 0)
        
        # Lexical diversity
        lexical_diversity = lexical.get('type_token_ratio', 0)
        
        return {
            'fluency': fluency,
            'coherence': coherence,
            'speech_rate': speech_rate,
            'pause_density': pause_density,
            'lexical_diversity': lexical_diversity,
            'hesitation_score': word_level.get('hesitation_frequency', 0),
            'rhythm_variability': word_level.get('word_duration_variability', 0)
        }
    
    def _update_real_time_metrics(self, metrics: Dict[str, float]):
        """Update real-time metrics arrays"""
        self.real_time_metrics['fluency'].append(metrics['fluency'])
        self.real_time_metrics['coherence'].append(metrics['coherence'])
        self.real_time_metrics['speech_rate'].append(metrics['speech_rate'])
        self.real_time_metrics['pause_density'].append(metrics['pause_density'])
    
    def _calculate_task_score(self) -> Dict[str, Any]:
        """Calculate final task score"""
        
        if not self.session_results:
            return {
                'task': self.current_task.task_type.value if self.current_task else 'unknown',
                'overall_score': 0,
                'status': 'no_data'
            }
        
        # Aggregate metrics
        all_metrics = [r['metrics'] for r in self.session_results]
        avg_metrics = {
            key: np.mean([m.get(key, 0) for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Task-specific scoring
        task_score = self._calculate_task_specific_score(avg_metrics)
        
        # Semantic relevance (if applicable)
        semantic_relevance = self._calculate_semantic_relevance()
        
        # Calculate overall cognitive score (0-100)
        cognitive_score = self._calculate_cognitive_score(
            task_score, 
            semantic_relevance, 
            avg_metrics
        )
        
        # Determine prediction
        if cognitive_score >= 75:
            prediction = "Healthy"
        elif cognitive_score >= 50:
            prediction = "Mild_Cognitive_Impairment"
        else:
            prediction = "Possible_Alzheimers"
        
        return {
            'task': self.current_task.task_type.value,
            'speech_fluency': avg_metrics['fluency'],
            'semantic_relevance': semantic_relevance,
            'pause_density': avg_metrics['pause_density'],
            'coherence': avg_metrics['coherence'],
            'lexical_diversity': avg_metrics['lexical_diversity'],
            'overall_score': cognitive_score,
            'prediction': prediction,
            'duration': time.time() - self.task_start_time,
            'segments_analyzed': len(self.session_results),
            'real_time_trends': {
                'fluency_trend': self._calculate_trend(self.real_time_metrics['fluency']),
                'coherence_trend': self._calculate_trend(self.real_time_metrics['coherence']),
                'speech_rate_trend': self._calculate_trend(self.real_time_metrics['speech_rate'])
            }
        }
    
    def _calculate_task_specific_score(self, metrics: Dict[str, float]) -> float:
        """Calculate task-specific score based on weights"""
        
        if not self.current_task:
            return 0.5
        
        weights = self.current_task.scoring_weights
        score = 0
        
        # Map metrics to scoring dimensions
        if 'fluency' in weights:
            score += weights['fluency'] * metrics.get('fluency', 0)
        
        if 'coherence' in weights:
            score += weights['coherence'] * metrics.get('coherence', 0)
        
        if 'relevance' in weights:
            # Will be calculated separately
            score += weights['relevance'] * 0.5  # Placeholder
        
        if 'accuracy' in weights:
            # For tasks like serial subtraction
            accuracy = self._calculate_accuracy()
            score += weights['accuracy'] * accuracy
        
        if 'quantity' in weights:
            # For verbal fluency
            quantity_score = min(1.0, len(self.session_results) / 10)
            score += weights['quantity'] * quantity_score
        
        return score
    
    def _calculate_semantic_relevance(self) -> float:
        """Calculate semantic relevance to task prompt"""
        
        if not self.session_results or not self.current_task:
            return 0.5
        
        # Combine all transcriptions
        all_text = " ".join([r.get('transcription', '') for r in self.session_results])
        
        if not all_text:
            return 0.0
        
        # Check for expected keywords
        expected = self.current_task.expected_keywords
        if expected:
            found = sum(1 for keyword in expected if keyword.lower() in all_text.lower())
            relevance = found / len(expected) if expected else 0
        else:
            # Use general coherence as proxy
            relevance = np.mean([r['metrics'].get('coherence', 0) for r in self.session_results])
        
        return min(1.0, relevance)
    
    def _calculate_accuracy(self) -> float:
        """Calculate accuracy for specific tasks"""
        
        if self.current_task.task_type == CognitiveTask.SERIAL_SUBTRACTION:
            # Check for correct sequence
            all_text = " ".join([r.get('transcription', '') for r in self.session_results])
            expected = ["100", "93", "86", "79", "72", "65", "58", "51", "44", "37", "30", "23", "16", "9", "2"]
            
            found_correct = 0
            for num in expected:
                if num in all_text:
                    found_correct += 1
            
            return found_correct / len(expected)
        
        return 0.5  # Default
    
    def _calculate_cognitive_score(self, task_score: float, semantic_relevance: float, 
                                  metrics: Dict[str, float]) -> float:
        """Calculate overall cognitive score (0-100)"""
        
        # Base score from task performance
        base_score = task_score * 40  # Max 40 points
        
        # Semantic relevance
        relevance_score = semantic_relevance * 20  # Max 20 points
        
        # Speech quality metrics
        fluency_score = metrics.get('fluency', 0) * 15  # Max 15 points
        coherence_score = metrics.get('coherence', 0) * 15  # Max 15 points
        
        # Penalty for excessive pausing
        pause_penalty = min(10, metrics.get('pause_density', 0) * 20)
        
        # Bonus for lexical diversity
        diversity_bonus = metrics.get('lexical_diversity', 0) * 10  # Max 10 points
        
        # Calculate total
        total_score = base_score + relevance_score + fluency_score + coherence_score + diversity_bonus - pause_penalty
        
        # Normalize to 0-100
        return max(0, min(100, total_score))
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(values))
        if len(values) > 0:
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 0.05:
                return "improving"
            elif slope < -0.05:
                return "declining"
        
        return "stable"
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics for dashboard"""
        
        return {
            'current_fluency': self.real_time_metrics['fluency'][-1] if self.real_time_metrics['fluency'] else 0,
            'current_coherence': self.real_time_metrics['coherence'][-1] if self.real_time_metrics['coherence'] else 0,
            'current_speech_rate': self.real_time_metrics['speech_rate'][-1] if self.real_time_metrics['speech_rate'] else 0,
            'current_pause_density': self.real_time_metrics['pause_density'][-1] if self.real_time_metrics['pause_density'] else 0,
            'fluency_history': self.real_time_metrics['fluency'][-10:],  # Last 10 values
            'coherence_history': self.real_time_metrics['coherence'][-10:],
            'speech_rate_history': self.real_time_metrics['speech_rate'][-10:],
            'segments_processed': len(self.session_results)
        }


def main():
    """Test the cognitive assessment system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cognitive Assessment System')
    parser.add_argument('--task', type=str, default='picture_description',
                       choices=['picture_description', 'story_recall', 'verbal_fluency', 'serial_subtraction'],
                       help='Type of cognitive task')
    parser.add_argument('--duration', type=int, default=30, help='Recording duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize system
    system = CognitiveAssessmentSystem()
    
    # Map task string to enum
    task_map = {
        'picture_description': CognitiveTask.PICTURE_DESCRIPTION,
        'story_recall': CognitiveTask.STORY_RECALL,
        'verbal_fluency': CognitiveTask.VERBAL_FLUENCY,
        'serial_subtraction': CognitiveTask.SERIAL_SUBTRACTION
    }
    
    task_type = task_map[args.task]
    
    # Start task
    task_info = system.start_task(task_type)
    print(f"\n{'='*60}")
    print(f"COGNITIVE ASSESSMENT: {task_type.value.upper()}")
    print(f"{'='*60}")
    print(f"\nPrompt: {task_info['prompt']}")
    print(f"Duration: {task_info['duration']} seconds")
    print(f"\nStarting recording in 3 seconds...")
    time.sleep(3)
    
    # Start recording
    system.start_real_time_recording()
    print("🎤 Recording... Speak now!")
    
    # Record for specified duration
    start_time = time.time()
    while time.time() - start_time < args.duration:
        # Get real-time metrics
        metrics = system.get_real_time_metrics()
        
        # Display current metrics
        print(f"\r⏱️ {int(time.time() - start_time)}s | "
              f"Fluency: {metrics['current_fluency']:.2f} | "
              f"Coherence: {metrics['current_coherence']:.2f} | "
              f"Speech Rate: {metrics['current_speech_rate']:.0f} wpm", end='')
        
        time.sleep(1)
    
    print("\n\n🛑 Recording stopped. Analyzing...")
    
    # Stop and get results
    results = system.stop_real_time_recording()
    
    # Display results
    print(f"\n{'='*60}")
    print("ASSESSMENT RESULTS")
    print(f"{'='*60}")
    print(f"Task: {results['task']}")
    print(f"Overall Score: {results['overall_score']:.1f}/100")
    print(f"Prediction: {results['prediction']}")
    print(f"\nDetailed Metrics:")
    print(f"  Speech Fluency: {results['speech_fluency']:.2%}")
    print(f"  Semantic Relevance: {results['semantic_relevance']:.2%}")
    print(f"  Pause Density: {results['pause_density']:.2%}")
    print(f"  Coherence: {results['coherence']:.2%}")
    print(f"  Lexical Diversity: {results['lexical_diversity']:.2%}")
    print(f"\nTrends:")
    print(f"  Fluency: {results['real_time_trends']['fluency_trend']}")
    print(f"  Coherence: {results['real_time_trends']['coherence_trend']}")
    print(f"  Speech Rate: {results['real_time_trends']['speech_rate_trend']}")
    print(f"\nDuration: {results['duration']:.1f} seconds")
    print(f"Segments Analyzed: {results['segments_analyzed']}")
    
    # Save results
    output_file = f"cognitive_assessment_{task_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
