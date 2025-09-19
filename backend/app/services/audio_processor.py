import librosa
import numpy as np
import soundfile as sf
import webrtcvad
import parselmouth
from parselmouth import praat
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import tempfile
import os

from app.core.config import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Core audio processing pipeline for voice biomarker extraction"""
    
    def __init__(self):
        self.sample_rate = settings.SAMPLE_RATE
        self.frame_length = settings.FRAME_LENGTH
        self.hop_length = settings.HOP_LENGTH
        self.n_mels = settings.N_MELS
        self.n_mfcc = settings.N_MFCC
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
    
    def validate_and_preprocess(self, audio_file_path: str) -> Tuple[np.ndarray, int]:
        """
        Validate and preprocess audio file
        Returns: (audio_data, sample_rate)
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            
            # Validate audio properties
            if len(audio_data) == 0:
                raise ValueError("Audio file is empty")
            
            if len(audio_data) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                raise ValueError("Audio file too short (minimum 0.5 seconds)")
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Remove silence from beginning and end
            audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
            
            logger.info(f"Preprocessed audio: {len(audio_data)/sr:.2f}s duration, {sr}Hz sample rate")
            
            return audio_data, sr
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise
    
    def voice_activity_detection(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Perform Voice Activity Detection using WebRTC VAD
        Returns: List of (start_time, end_time) tuples for speech segments
        """
        try:
            # Convert to 16-bit PCM for WebRTC VAD
            audio_16bit = (audio_data * 32767).astype(np.int16)
            
            # Frame size for VAD (10ms, 20ms, or 30ms)
            frame_duration_ms = 20
            frame_size = int(sample_rate * frame_duration_ms / 1000)
            
            speech_segments = []
            current_segment_start = None
            
            for i in range(0, len(audio_16bit) - frame_size, frame_size):
                frame = audio_16bit[i:i + frame_size].tobytes()
                
                if len(frame) == frame_size * 2:  # 2 bytes per sample
                    is_speech = self.vad.is_speech(frame, sample_rate)
                    
                    timestamp = i / sample_rate
                    
                    if is_speech and current_segment_start is None:
                        current_segment_start = timestamp
                    elif not is_speech and current_segment_start is not None:
                        speech_segments.append((current_segment_start, timestamp))
                        current_segment_start = None
            
            # Close final segment if needed
            if current_segment_start is not None:
                speech_segments.append((current_segment_start, len(audio_16bit) / sample_rate))
            
            logger.info(f"VAD detected {len(speech_segments)} speech segments")
            return speech_segments
            
        except Exception as e:
            logger.error(f"VAD failed: {str(e)}")
            return [(0, len(audio_data) / sample_rate)]  # Return full audio as fallback
    
    def extract_acoustic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract comprehensive acoustic features using Parselmouth/Praat
        """
        try:
            # Create Parselmouth Sound object
            sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
            
            features = {}
            
            # Pitch analysis
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
            else:
                features['pitch_mean'] = None
                features['pitch_std'] = None
                features['pitch_range'] = None
            
            # Jitter analysis
            try:
                point_process = praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
                jitter_local = praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                jitter_rap = praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                
                features['jitter_local'] = float(jitter_local) if not np.isnan(jitter_local) else None
                features['jitter_rap'] = float(jitter_rap) if not np.isnan(jitter_rap) else None
            except:
                features['jitter_local'] = None
                features['jitter_rap'] = None
            
            # Shimmer analysis
            try:
                shimmer_local = praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq3 = praat.call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                
                features['shimmer_local'] = float(shimmer_local) if not np.isnan(shimmer_local) else None
                features['shimmer_apq3'] = float(shimmer_apq3) if not np.isnan(shimmer_apq3) else None
            except:
                features['shimmer_local'] = None
                features['shimmer_apq3'] = None
            
            # Harmonics-to-Noise Ratio (HNR)
            try:
                harmonicity = sound.to_harmonicity(time_step=0.01, minimum_pitch=75, silence_threshold=0.1, periods_per_window=1.0)
                hnr_values = harmonicity.values[harmonicity.values != -200]  # Remove undefined values
                
                if len(hnr_values) > 0:
                    features['hnr_mean'] = float(np.mean(hnr_values))
                    features['hnr_std'] = float(np.std(hnr_values))
                else:
                    features['hnr_mean'] = None
                    features['hnr_std'] = None
            except:
                features['hnr_mean'] = None
                features['hnr_std'] = None
            
            # Formant analysis
            try:
                formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500, window_length=0.025, pre_emphasis_from=50)
                
                f1_values = []
                f2_values = []
                f3_values = []
                
                for i in range(formants.get_number_of_frames()):
                    f1 = formants.get_value_at_time(1, formants.get_time_from_frame_number(i + 1))
                    f2 = formants.get_value_at_time(2, formants.get_time_from_frame_number(i + 1))
                    f3 = formants.get_value_at_time(3, formants.get_time_from_frame_number(i + 1))
                    
                    if not np.isnan(f1): f1_values.append(f1)
                    if not np.isnan(f2): f2_values.append(f2)
                    if not np.isnan(f3): f3_values.append(f3)
                
                features['f1_mean'] = float(np.mean(f1_values)) if f1_values else None
                features['f2_mean'] = float(np.mean(f2_values)) if f2_values else None
                features['f3_mean'] = float(np.mean(f3_values)) if f3_values else None
                
                # Formant dispersion
                if f1_values and f2_values and f3_values:
                    formant_means = [np.mean(f1_values), np.mean(f2_values), np.mean(f3_values)]
                    features['formant_dispersion'] = float(np.std(formant_means))
                else:
                    features['formant_dispersion'] = None
                    
            except:
                features['f1_mean'] = None
                features['f2_mean'] = None
                features['f3_mean'] = None
                features['formant_dispersion'] = None
            
            logger.info("Acoustic feature extraction completed")
            return features
            
        except Exception as e:
            logger.error(f"Acoustic feature extraction failed: {str(e)}")
            return {}
    
    def extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract spectral features using librosa
        """
        try:
            features = {}
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio_data, frame_length=self.frame_length, hop_length=self.hop_length
            )[0]
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, sr=sample_rate, n_mfcc=self.n_mfcc,
                hop_length=self.hop_length, n_fft=self.frame_length
            )
            
            # Store mean and std of each MFCC coefficient
            mfcc_features = {}
            for i in range(self.n_mfcc):
                mfcc_features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                mfcc_features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            features['mfcc_features'] = mfcc_features
            
            logger.info("Spectral feature extraction completed")
            return features
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {str(e)}")
            return {}
    
    def assess_voice_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Assess voice quality metrics (breathiness, hoarseness, tremor)
        Based on Saeedi et al. (2024) voice quality markers
        """
        try:
            features = {}
            
            # Signal quality assessment
            # SNR estimation
            signal_power = np.mean(audio_data ** 2)
            noise_floor = np.percentile(np.abs(audio_data), 10)  # Estimate noise floor
            snr_estimate = 10 * np.log10(signal_power / (noise_floor ** 2 + 1e-10))
            features['signal_quality_score'] = float(max(0, min(1, (snr_estimate + 10) / 40)))  # Normalize to 0-1
            
            # Noise level
            features['noise_level'] = float(noise_floor)
            
            # Clipping detection
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
            features['clipping_detected'] = bool(clipped_samples > len(audio_data) * 0.001)  # >0.1% clipped
            
            # Voice quality metrics using spectral analysis
            # Breathiness: High-frequency noise relative to harmonic content
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            
            # Harmonic-to-noise ratio in frequency domain
            harmonic_bins = magnitude[:magnitude.shape[0]//4]  # Lower frequencies (harmonics)
            noise_bins = magnitude[magnitude.shape[0]//2:]     # Higher frequencies (noise)
            
            harmonic_energy = np.mean(harmonic_bins ** 2)
            noise_energy = np.mean(noise_bins ** 2)
            
            breathiness_score = noise_energy / (harmonic_energy + 1e-10)
            features['breathiness_score'] = float(min(1.0, breathiness_score))
            
            # Hoarseness: Irregularity in pitch and amplitude
            # Use jitter and shimmer as proxies (already computed in acoustic features)
            features['hoarseness_score'] = None  # Will be computed from jitter/shimmer
            
            # Vocal tremor: Low-frequency modulation
            # Analyze pitch contour for tremor-like oscillations (4-12 Hz)
            try:
                sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
                pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
                pitch_values = pitch.selected_array['frequency']
                pitch_values = pitch_values[pitch_values != 0]
                
                if len(pitch_values) > 50:  # Need sufficient data
                    # Detrend pitch contour
                    detrended = pitch_values - np.mean(pitch_values)
                    
                    # FFT to find tremor frequencies
                    fft = np.fft.fft(detrended)
                    freqs = np.fft.fftfreq(len(detrended), d=0.01)  # 0.01s time step
                    
                    # Look for peaks in 4-12 Hz range
                    tremor_mask = (freqs >= 4) & (freqs <= 12)
                    tremor_power = np.sum(np.abs(fft[tremor_mask]) ** 2)
                    total_power = np.sum(np.abs(fft) ** 2)
                    
                    features['vocal_tremor_score'] = float(tremor_power / (total_power + 1e-10))
                else:
                    features['vocal_tremor_score'] = None
                    
            except:
                features['vocal_tremor_score'] = None
            
            logger.info("Voice quality assessment completed")
            return features
            
        except Exception as e:
            logger.error(f"Voice quality assessment failed: {str(e)}")
            return {}
    
    def extract_timing_features(self, speech_segments: List[Tuple[float, float]], 
                              total_duration: float) -> Dict[str, Any]:
        """
        Extract speech timing and fluency features
        Based on Yang et al. (2022) timing measures
        """
        try:
            features = {}
            
            if not speech_segments:
                return features
            
            # Calculate pause statistics
            pauses = []
            for i in range(len(speech_segments) - 1):
                pause_start = speech_segments[i][1]
                pause_end = speech_segments[i + 1][0]
                pause_duration = pause_end - pause_start
                if pause_duration > 0.1:  # Minimum pause threshold
                    pauses.append(pause_duration)
            
            # Speech timing features
            total_speech_time = sum(end - start for start, end in speech_segments)
            total_pause_time = sum(pauses) if pauses else 0
            
            features['speech_rate_syllables_per_second'] = None  # Requires syllable counting from transcript
            features['articulation_rate'] = None  # Requires syllable counting
            features['pause_frequency'] = float(len(pauses) / (total_duration / 60))  # Pauses per minute
            features['mean_pause_duration'] = float(np.mean(pauses)) if pauses else 0.0
            
            # Additional timing metrics
            features['speech_time_ratio'] = float(total_speech_time / total_duration)
            features['pause_time_ratio'] = float(total_pause_time / total_duration)
            
            logger.info("Timing feature extraction completed")
            return features
            
        except Exception as e:
            logger.error(f"Timing feature extraction failed: {str(e)}")
            return {}
    
    def process_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Complete audio processing pipeline
        """
        try:
            logger.info(f"Starting audio processing for: {audio_file_path}")
            
            # Step 1: Validate and preprocess
            audio_data, sample_rate = self.validate_and_preprocess(audio_file_path)
            total_duration = len(audio_data) / sample_rate
            
            # Step 2: Voice Activity Detection
            speech_segments = self.voice_activity_detection(audio_data, sample_rate)
            
            # Step 3: Extract acoustic features
            acoustic_features = self.extract_acoustic_features(audio_data, sample_rate)
            
            # Step 4: Extract spectral features
            spectral_features = self.extract_spectral_features(audio_data, sample_rate)
            
            # Step 5: Assess voice quality
            voice_quality = self.assess_voice_quality(audio_data, sample_rate)
            
            # Step 6: Extract timing features
            timing_features = self.extract_timing_features(speech_segments, total_duration)
            
            # Combine all features
            all_features = {
                **acoustic_features,
                **spectral_features,
                **voice_quality,
                **timing_features,
                'total_duration_seconds': float(total_duration),
                'speech_segments': speech_segments,
                'processing_metadata': {
                    'sample_rate': sample_rate,
                    'frame_length': self.frame_length,
                    'hop_length': self.hop_length,
                    'n_speech_segments': len(speech_segments)
                }
            }
            
            logger.info("Audio processing completed successfully")
            return all_features
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise
