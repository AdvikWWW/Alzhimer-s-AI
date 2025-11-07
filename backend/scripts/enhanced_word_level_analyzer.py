#!/usr/bin/env python3
"""
Enhanced Word-Level Analyzer for Alzheimer's Detection
Implements advanced linguistic and acoustic features with word-by-word analysis
Uses Wav2Vec2, Whisper, and advanced speech processing
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WhisperProcessor, WhisperForConditionalGeneration
from scipy.stats import entropy
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WordLevelAnalyzer:
    """Advanced word-level speech analysis with deep learning embeddings"""
    
    def __init__(self, use_gpu: bool = True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.wav2vec_processor = None
        self.wav2vec_model = None
        self.whisper_processor = None
        self.whisper_model = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Wav2Vec2 and Whisper models"""
        try:
            # Wav2Vec2 for speech embeddings
            logger.info("Loading Wav2Vec2 model...")
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            self.wav2vec_model.eval()
            logger.info("✓ Wav2Vec2 loaded")
            
        except Exception as e:
            logger.warning(f"Failed to load Wav2Vec2: {str(e)}")
    
    def analyze_audio_word_by_word(self, audio_path: str, word_timestamps: List[Dict]) -> Dict[str, Any]:
        """
        Analyze each word with advanced acoustic and linguistic features
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            word_analyses = []
            
            for i, word_info in enumerate(word_timestamps):
                word = word_info.get('word', '').strip()
                start_time = word_info.get('start', 0)
                end_time = word_info.get('end', 0)
                
                if not word or end_time <= start_time:
                    continue
                
                # Extract word audio segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                word_audio = audio[start_sample:end_sample]
                
                if len(word_audio) < 100:  # Too short
                    continue
                
                # Analyze this word
                word_analysis = self._analyze_single_word(
                    word_audio, sr, word, start_time, end_time, i
                )
                word_analyses.append(word_analysis)
            
            # Aggregate word-level features
            aggregated_features = self._aggregate_word_features(word_analyses)
            
            return {
                'word_analyses': word_analyses,
                'aggregated_features': aggregated_features,
                'num_words_analyzed': len(word_analyses)
            }
            
        except Exception as e:
            logger.error(f"Word-by-word analysis failed: {str(e)}")
            return {'word_analyses': [], 'aggregated_features': {}, 'num_words_analyzed': 0}
    
    def _analyze_single_word(self, word_audio: np.ndarray, sr: int, word: str, 
                            start_time: float, end_time: float, word_index: int) -> Dict[str, Any]:
        """Comprehensive analysis of a single word"""
        
        analysis = {
            'word': word,
            'word_index': word_index,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
        
        # 1. Acoustic features
        acoustic = self._extract_word_acoustic_features(word_audio, sr)
        analysis.update(acoustic)
        
        # 2. MFCC deltas
        mfcc_features = self._extract_mfcc_deltas(word_audio, sr)
        analysis.update(mfcc_features)
        
        # 3. Spectral entropy
        spectral_features = self._extract_spectral_features(word_audio, sr)
        analysis.update(spectral_features)
        
        # 4. Formant tracking
        formant_features = self._extract_formant_dynamics(word_audio, sr)
        analysis.update(formant_features)
        
        # 5. Wav2Vec2 embeddings
        if self.wav2vec_model is not None:
            embedding_features = self._extract_wav2vec_embeddings(word_audio, sr)
            analysis.update(embedding_features)
        
        return analysis
    
    def _extract_word_acoustic_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract acoustic features for a word"""
        features = {}
        
        try:
            # Energy
            features['energy'] = float(np.sum(audio ** 2))
            features['rms_energy'] = float(np.sqrt(np.mean(audio ** 2)))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
            
        except Exception as e:
            logger.debug(f"Acoustic feature extraction failed: {str(e)}")
        
        return features
    
    def _extract_mfcc_deltas(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract MFCC coefficients and their deltas"""
        features = {}
        
        try:
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Delta MFCCs (velocity)
            delta_mfccs = librosa.feature.delta(mfccs)
            
            # Delta-delta MFCCs (acceleration)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Statistics
            features['mfcc_mean'] = float(np.mean(mfccs))
            features['mfcc_std'] = float(np.std(mfccs))
            features['delta_mfcc_mean'] = float(np.mean(delta_mfccs))
            features['delta_mfcc_std'] = float(np.std(delta_mfccs))
            features['delta2_mfcc_mean'] = float(np.mean(delta2_mfccs))
            features['delta2_mfcc_std'] = float(np.std(delta2_mfccs))
            
        except Exception as e:
            logger.debug(f"MFCC delta extraction failed: {str(e)}")
        
        return features
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features including entropy"""
        features = {}
        
        try:
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(centroid))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth'] = float(np.mean(bandwidth))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff'] = float(np.mean(rolloff))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            
            # Spectral entropy
            stft = np.abs(librosa.stft(audio))
            spectral_entropy_values = []
            for frame in stft.T:
                if np.sum(frame) > 0:
                    normalized = frame / np.sum(frame)
                    spectral_entropy_values.append(entropy(normalized))
            
            if spectral_entropy_values:
                features['spectral_entropy_mean'] = float(np.mean(spectral_entropy_values))
                features['spectral_entropy_std'] = float(np.std(spectral_entropy_values))
            
        except Exception as e:
            logger.debug(f"Spectral feature extraction failed: {str(e)}")
        
        return features
    
    def _extract_formant_dynamics(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract formant dynamics and shifts"""
        features = {}
        
        try:
            # Use LPC to estimate formants
            # Pre-emphasis
            pre_emphasized = librosa.effects.preemphasis(audio)
            
            # Frame the signal
            frames = librosa.util.frame(pre_emphasized, frame_length=int(0.025 * sr), hop_length=int(0.010 * sr))
            
            formant_tracks = []
            
            for frame in frames.T:
                if len(frame) > 0 and np.std(frame) > 0.01:
                    # LPC analysis
                    try:
                        lpc_order = 12
                        a = librosa.lpc(frame, order=lpc_order)
                        
                        # Find roots
                        roots = np.roots(a)
                        roots = roots[np.imag(roots) >= 0]
                        
                        # Convert to frequencies
                        angles = np.arctan2(np.imag(roots), np.real(roots))
                        freqs = sorted(angles * (sr / (2 * np.pi)))
                        
                        # Take first 3 formants
                        formants = [f for f in freqs if 90 < f < 5000][:3]
                        if len(formants) >= 2:
                            formant_tracks.append(formants)
                    except:
                        continue
            
            if formant_tracks:
                formant_tracks = np.array(formant_tracks)
                
                # Formant statistics
                for i in range(min(3, formant_tracks.shape[1])):
                    formant_values = formant_tracks[:, i]
                    features[f'formant_{i+1}_mean'] = float(np.mean(formant_values))
                    features[f'formant_{i+1}_std'] = float(np.std(formant_values))
                    
                    # Formant shift (rate of change)
                    if len(formant_values) > 1:
                        shifts = np.diff(formant_values)
                        features[f'formant_{i+1}_shift_mean'] = float(np.mean(np.abs(shifts)))
        
        except Exception as e:
            logger.debug(f"Formant extraction failed: {str(e)}")
        
        return features
    
    def _extract_wav2vec_embeddings(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract Wav2Vec2 embeddings for semantic representation"""
        features = {}
        
        try:
            # Resample if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Process audio
            inputs = self.wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
            
            # Aggregate embeddings
            features['wav2vec_embedding_mean'] = float(np.mean(embeddings))
            features['wav2vec_embedding_std'] = float(np.std(embeddings))
            features['wav2vec_embedding_max'] = float(np.max(embeddings))
            features['wav2vec_embedding_min'] = float(np.min(embeddings))
            
        except Exception as e:
            logger.debug(f"Wav2Vec2 embedding extraction failed: {str(e)}")
        
        return features
    
    def _aggregate_word_features(self, word_analyses: List[Dict]) -> Dict[str, float]:
        """Aggregate word-level features into session-level features"""
        
        if not word_analyses:
            return {}
        
        aggregated = {}
        
        # Collect all numeric features
        feature_keys = set()
        for analysis in word_analyses:
            for key, value in analysis.items():
                if isinstance(value, (int, float)) and key not in ['word_index', 'start_time', 'end_time']:
                    feature_keys.add(key)
        
        # Aggregate each feature
        for key in feature_keys:
            values = [a[key] for a in word_analyses if key in a and isinstance(a[key], (int, float))]
            
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                aggregated[f'{key}_median'] = float(np.median(values))
                aggregated[f'{key}_max'] = float(np.max(values))
                aggregated[f'{key}_min'] = float(np.min(values))
        
        # Word timing statistics
        durations = [a['duration'] for a in word_analyses if 'duration' in a]
        if durations:
            aggregated['word_duration_variability'] = float(np.std(durations) / (np.mean(durations) + 1e-10))
        
        # Inter-word pause analysis
        pauses = []
        for i in range(len(word_analyses) - 1):
            pause = word_analyses[i+1]['start_time'] - word_analyses[i]['end_time']
            if pause > 0:
                pauses.append(pause)
        
        if pauses:
            aggregated['inter_word_pause_mean'] = float(np.mean(pauses))
            aggregated['inter_word_pause_std'] = float(np.std(pauses))
            aggregated['long_pauses_ratio'] = float(sum(1 for p in pauses if p > 0.5) / len(pauses))
        
        return aggregated


class IntelligentAlzheimerScorer:
    """
    Intelligent scoring system combining acoustic and linguistic biomarkers
    """
    
    def __init__(self):
        # Clinical thresholds based on research
        self.thresholds = {
            'pause_rate_high': 0.25,  # >25% pause time indicates risk
            'speech_rate_low': 110,   # <110 words/min indicates risk
            'vocabulary_diversity_low': 0.60,  # <60% TTR indicates risk
            'hesitation_high': 0.15,  # >15% hesitation rate indicates risk
            'pitch_variability_low': 20,  # Low pitch range indicates monotone speech
            'spectral_entropy_low': 2.0,  # Low entropy indicates reduced complexity
        }
    
    def score_recording(self, features: Dict[str, Any], transcription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate intelligent Alzheimer's risk score
        """
        
        scores = {
            'acoustic_biomarkers': {},
            'linguistic_biomarkers': {},
            'cognitive_biomarkers': {},
            'overall_score': 0.0,
            'risk_category': 'Unknown',
            'confidence': 0.0
        }
        
        # 1. Acoustic biomarkers
        acoustic_score = self._score_acoustic_biomarkers(features)
        scores['acoustic_biomarkers'] = acoustic_score
        
        # 2. Linguistic biomarkers
        linguistic_score = self._score_linguistic_biomarkers(features, transcription)
        scores['linguistic_biomarkers'] = linguistic_score
        
        # 3. Cognitive biomarkers
        cognitive_score = self._score_cognitive_biomarkers(features, transcription)
        scores['cognitive_biomarkers'] = cognitive_score
        
        # 4. Combined score
        overall_score = (
            acoustic_score['score'] * 0.35 +
            linguistic_score['score'] * 0.35 +
            cognitive_score['score'] * 0.30
        )
        
        scores['overall_score'] = overall_score
        
        # 5. Risk categorization
        if overall_score < 0.3:
            scores['risk_category'] = 'Low_Risk_Healthy'
        elif overall_score < 0.6:
            scores['risk_category'] = 'Moderate_Risk_Uncertain'
        else:
            scores['risk_category'] = 'High_Risk_Possible_Alzheimers'
        
        # 6. Confidence based on feature completeness
        feature_completeness = sum([
            acoustic_score.get('completeness', 0),
            linguistic_score.get('completeness', 0),
            cognitive_score.get('completeness', 0)
        ]) / 3
        
        scores['confidence'] = feature_completeness
        
        return scores
    
    def _score_acoustic_biomarkers(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score acoustic biomarkers"""
        
        score_components = []
        indicators = []
        
        # Pitch variability
        pitch_range = features.get('pitch_range', 0)
        if pitch_range < self.thresholds['pitch_variability_low']:
            score_components.append(0.7)
            indicators.append('Reduced pitch variability (monotone speech)')
        else:
            score_components.append(0.2)
        
        # Voice quality (jitter, shimmer)
        jitter = features.get('jitter_local', 0)
        if jitter > 0.01:  # High jitter indicates voice instability
            score_components.append(0.6)
            indicators.append('Increased voice jitter')
        else:
            score_components.append(0.3)
        
        # Spectral entropy
        spectral_entropy = features.get('spectral_entropy_mean', 3.0)
        if spectral_entropy < self.thresholds['spectral_entropy_low']:
            score_components.append(0.6)
            indicators.append('Low spectral complexity')
        else:
            score_components.append(0.2)
        
        # Pause characteristics
        pause_ratio = features.get('pause_time_ratio', 0)
        if pause_ratio > self.thresholds['pause_rate_high']:
            score_components.append(0.8)
            indicators.append('Excessive pausing')
        else:
            score_components.append(0.2)
        
        score = np.mean(score_components) if score_components else 0.5
        completeness = len(score_components) / 4.0
        
        return {
            'score': score,
            'indicators': indicators,
            'completeness': completeness
        }
    
    def _score_linguistic_biomarkers(self, features: Dict[str, Any], transcription: Dict[str, Any]) -> Dict[str, Any]:
        """Score linguistic biomarkers"""
        
        score_components = []
        indicators = []
        
        # Vocabulary diversity
        ttr = features.get('type_token_ratio', 0.7)
        if ttr < self.thresholds['vocabulary_diversity_low']:
            score_components.append(0.7)
            indicators.append('Reduced vocabulary diversity')
        else:
            score_components.append(0.2)
        
        # Word-finding difficulty (filled pauses, repetitions)
        filled_pause_rate = features.get('filled_pause_rate', 0)
        if filled_pause_rate > 0.10:
            score_components.append(0.6)
            indicators.append('Frequent filled pauses (um, uh)')
        else:
            score_components.append(0.2)
        
        # Repetitions
        repetition_rate = features.get('repetition_rate', 0)
        if repetition_rate > 0.05:
            score_components.append(0.6)
            indicators.append('Word repetitions detected')
        else:
            score_components.append(0.2)
        
        # Sentence complexity
        mean_sentence_length = features.get('mean_sentence_length', 10)
        if mean_sentence_length < 6:
            score_components.append(0.6)
            indicators.append('Simplified sentence structure')
        else:
            score_components.append(0.2)
        
        score = np.mean(score_components) if score_components else 0.5
        completeness = len(score_components) / 4.0
        
        return {
            'score': score,
            'indicators': indicators,
            'completeness': completeness
        }
    
    def _score_cognitive_biomarkers(self, features: Dict[str, Any], transcription: Dict[str, Any]) -> Dict[str, Any]:
        """Score cognitive biomarkers"""
        
        score_components = []
        indicators = []
        
        # Speech fluency
        speech_rate = features.get('speech_rate_syllables_per_second', 2.5)
        if speech_rate and speech_rate < 1.8:  # ~110 words/min
            score_components.append(0.7)
            indicators.append('Slow speech rate')
        else:
            score_components.append(0.2)
        
        # Semantic coherence
        coherence = features.get('semantic_coherence_score', 0.7)
        if coherence < 0.5:
            score_components.append(0.8)
            indicators.append('Reduced semantic coherence')
        else:
            score_components.append(0.2)
        
        # Idea density
        idea_density = features.get('idea_density', 0.5)
        if idea_density < 0.4:
            score_components.append(0.7)
            indicators.append('Low idea density')
        else:
            score_components.append(0.2)
        
        # Long pauses (hesitation)
        long_pause_ratio = features.get('long_pauses_ratio', 0)
        if long_pause_ratio > self.thresholds['hesitation_high']:
            score_components.append(0.7)
            indicators.append('Frequent long pauses (hesitation)')
        else:
            score_components.append(0.2)
        
        score = np.mean(score_components) if score_components else 0.5
        completeness = len(score_components) / 4.0
        
        return {
            'score': score,
            'indicators': indicators,
            'completeness': completeness
        }


def main():
    """Test the enhanced analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced word-level Alzheimer\'s voice analysis')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--output', type=str, default='word_analysis.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    logger.info("Initializing Enhanced Word-Level Analyzer...")
    analyzer = WordLevelAnalyzer(use_gpu=True)
    scorer = IntelligentAlzheimerScorer()
    
    # For demo, create dummy word timestamps
    # In production, these would come from ASR service
    dummy_timestamps = [
        {'word': 'hello', 'start': 0.0, 'end': 0.5},
        {'word': 'world', 'start': 0.6, 'end': 1.0},
    ]
    
    logger.info(f"Analyzing: {args.audio}")
    results = analyzer.analyze_audio_word_by_word(args.audio, dummy_timestamps)
    
    logger.info(f"Analyzed {results['num_words_analyzed']} words")
    logger.info(f"Extracted {len(results['aggregated_features'])} aggregated features")
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
