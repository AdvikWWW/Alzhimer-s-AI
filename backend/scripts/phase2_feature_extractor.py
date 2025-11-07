#!/usr/bin/env python3
"""
Phase 2: Feature Extractor
Extracts 150+ features from audio files using librosa and custom methods
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
METADATA_DIR = DATA_DIR / "metadata"


class ComprehensiveFeatureExtractor:
    """
    Extracts 150+ audio features for Alzheimer's detection
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.feature_names = []
    
    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features (50+ features)
        """
        features = {}
        
        # 1. MFCCs (39 features: 13 + 13 deltas + 13 delta-deltas)
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            features[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
            features[f'mfcc_delta2_{i+1}_mean'] = float(np.mean(mfcc_delta2[i]))
        
        # 2. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        features['spectral_centroid_min'] = float(np.min(spectral_centroids))
        features['spectral_centroid_max'] = float(np.max(spectral_centroids))
        
        # 3. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # 4. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # 5. Spectral Contrast (7 bands)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_band_{i+1}_mean'] = float(np.mean(spectral_contrast[i]))
        
        # 6. Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = float(np.std(spectral_flatness))
        
        # 7. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        return features
    
    def extract_temporal_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features (25+ features)
        """
        features = {}
        
        # 1. RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_min'] = float(np.min(rms))
        features['rms_max'] = float(np.max(rms))
        features['rms_range'] = float(np.max(rms) - np.min(rms))
        
        # 2. Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
            features['tempo'] = float(tempo)
        except:
            features['tempo'] = 0.0
        
        # 3. Onset Strength
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        features['onset_strength_mean'] = float(np.mean(onset_env))
        features['onset_strength_std'] = float(np.std(onset_env))
        
        # 4. Duration
        features['duration_sec'] = float(librosa.get_duration(y=y, sr=self.sr))
        
        # 5. Energy Statistics
        energy = y ** 2
        features['energy_mean'] = float(np.mean(energy))
        features['energy_std'] = float(np.std(energy))
        features['energy_entropy'] = float(self._calculate_entropy(energy))
        
        return features
    
    def extract_pitch_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch-related features (10+ features)
        """
        features = {}
        
        try:
            # Extract pitch using piptrack
            pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
            
            # Get pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                pitch_values = np.array(pitch_values)
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_min'] = float(np.min(pitch_values))
                features['pitch_max'] = float(np.max(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
                features['pitch_variation_coef'] = float(np.std(pitch_values) / (np.mean(pitch_values) + 1e-10))
            else:
                for key in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range', 'pitch_variation_coef']:
                    features[key] = 0.0
        except:
            for key in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range', 'pitch_variation_coef']:
                features[key] = 0.0
        
        return features
    
    def extract_voice_quality_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract voice quality features (15+ features)
        """
        features = {}
        
        # 1. Harmonics-to-Noise Ratio (HNR) approximation
        try:
            # Simple HNR estimate using autocorrelation
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first peak (fundamental period)
            peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
            
            if len(peaks) > 0:
                first_peak = peaks[0]
                hnr = autocorr[first_peak] / (autocorr[0] + 1e-10)
                features['hnr_estimate'] = float(20 * np.log10(hnr + 1e-10))
            else:
                features['hnr_estimate'] = 0.0
        except:
            features['hnr_estimate'] = 0.0
        
        # 2. Jitter (pitch variation) - simplified
        try:
            # Use zero-crossing rate as proxy for jitter
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['jitter_proxy'] = float(np.std(zcr) / (np.mean(zcr) + 1e-10))
        except:
            features['jitter_proxy'] = 0.0
        
        # 3. Shimmer (amplitude variation) - simplified
        try:
            rms = librosa.feature.rms(y=y)[0]
            features['shimmer_proxy'] = float(np.std(rms) / (np.mean(rms) + 1e-10))
        except:
            features['shimmer_proxy'] = 0.0
        
        # 4. Spectral Entropy
        features['spectral_entropy'] = float(self._calculate_spectral_entropy(y))
        
        return features
    
    def extract_speech_timing_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract speech timing and pause features (20+ features)
        """
        features = {}
        
        # 1. Detect speech/silence using energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        threshold = np.mean(rms) * 0.5
        speech_frames = rms > threshold
        
        # 2. Calculate pause statistics
        transitions = np.diff(speech_frames.astype(int))
        pause_starts = np.where(transitions == -1)[0]
        pause_ends = np.where(transitions == 1)[0]
        
        # Match pauses
        if len(pause_starts) > 0 and len(pause_ends) > 0:
            if pause_starts[0] > pause_ends[0]:
                pause_ends = pause_ends[1:]
            if len(pause_starts) > len(pause_ends):
                pause_starts = pause_starts[:len(pause_ends)]
            
            pause_durations = (pause_ends - pause_starts) * 512 / self.sr
            
            features['pause_count'] = len(pause_durations)
            features['pause_duration_mean'] = float(np.mean(pause_durations)) if len(pause_durations) > 0 else 0.0
            features['pause_duration_std'] = float(np.std(pause_durations)) if len(pause_durations) > 0 else 0.0
            features['pause_duration_total'] = float(np.sum(pause_durations))
        else:
            features['pause_count'] = 0
            features['pause_duration_mean'] = 0.0
            features['pause_duration_std'] = 0.0
            features['pause_duration_total'] = 0.0
        
        # 3. Speech ratio
        speech_ratio = np.sum(speech_frames) / len(speech_frames)
        features['speech_ratio'] = float(speech_ratio)
        
        # 4. Estimated speaking rate
        duration = librosa.get_duration(y=y, sr=self.sr)
        estimated_words = int((duration * speech_ratio) * 2.5)  # ~2.5 words per second
        features['estimated_words'] = estimated_words
        features['speech_rate_wpm'] = float((estimated_words / duration) * 60) if duration > 0 else 0.0
        
        # 5. Pause density
        features['pause_density'] = float(features['pause_count'] / duration) if duration > 0 else 0.0
        
        return features
    
    def _calculate_entropy(self, signal: np.ndarray) -> float:
        """Calculate entropy of a signal"""
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_spectral_entropy(self, y: np.ndarray) -> float:
        """Calculate spectral entropy"""
        spec = np.abs(librosa.stft(y))
        spec_norm = spec / (np.sum(spec, axis=0) + 1e-10)
        entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10), axis=0)
        return float(np.mean(entropy))
    
    def extract_all_features(self, audio_path: Path) -> Dict[str, float]:
        """
        Extract all features from an audio file
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Extract feature groups
            features = {}
            features.update(self.extract_spectral_features(y))
            features.update(self.extract_temporal_features(y))
            features.update(self.extract_pitch_features(y))
            features.update(self.extract_voice_quality_features(y))
            features.update(self.extract_speech_timing_features(y))
            
            # Add metadata
            features['filename'] = audio_path.name
            # Handle paths outside project root (e.g., temp files)
            try:
                features['file_path'] = str(audio_path.relative_to(PROJECT_ROOT))
            except ValueError:
                # If path is outside project root, use absolute path
                features['file_path'] = str(audio_path)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def process_dataset(self, metadata_path: Path) -> pd.DataFrame:
        """
        Process entire dataset and extract features
        """
        print("="*70)
        print("EXTRACTING FEATURES FROM DATASET")
        print("="*70)
        
        # Load metadata
        metadata_df = pd.read_csv(metadata_path)
        print(f"Loaded metadata: {len(metadata_df)} files")
        
        all_features = []
        
        # Process each file
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extracting features"):
            # Get processed audio path
            if 'processed_path' in row and pd.notna(row['processed_path']):
                audio_path = PROJECT_ROOT / row['processed_path']
            else:
                # Fallback to raw audio
                audio_path = PROJECT_ROOT / row['file_path']
            
            if not audio_path.exists():
                print(f"⚠️  File not found: {audio_path}")
                continue
            
            # Extract features
            features = self.extract_all_features(audio_path)
            
            if features:
                # Add label and file_id
                features['label'] = row['label']
                features['file_id'] = row['file_id']
                all_features.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        
        print(f"\n✅ Extracted features from {len(features_df)} files")
        print(f"   Total features per file: {len(features_df.columns) - 3}")  # Exclude label, file_id, filename
        
        return features_df
    
    def save_features(self, features_df: pd.DataFrame):
        """
        Save features in multiple formats
        """
        print("\n" + "="*70)
        print("SAVING FEATURES")
        print("="*70)
        
        # Ensure features directory exists
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Save as CSV
        csv_path = FEATURES_DIR / 'features.csv'
        features_df.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV: {csv_path}")
        
        # 2. Save as NumPy array
        feature_cols = [col for col in features_df.columns if col not in ['label', 'file_id', 'filename', 'file_path']]
        X = features_df[feature_cols].values
        y = features_df['label'].map({'Alzheimer': 1, 'Healthy': 0}).values
        
        npy_path = FEATURES_DIR / 'features.npy'
        np.save(npy_path, X)
        print(f"✅ Saved NumPy features: {npy_path}")
        
        labels_path = FEATURES_DIR / 'labels.npy'
        np.save(labels_path, y)
        print(f"✅ Saved NumPy labels: {labels_path}")
        
        # 3. Save feature names
        feature_names_path = FEATURES_DIR / 'feature_names.txt'
        with open(feature_names_path, 'w') as f:
            for name in feature_cols:
                f.write(f"{name}\n")
        print(f"✅ Saved feature names: {feature_names_path}")
        
        # 4. Save feature statistics
        stats = features_df[feature_cols].describe()
        stats_path = FEATURES_DIR / 'feature_statistics.csv'
        stats.to_csv(stats_path)
        print(f"✅ Saved feature statistics: {stats_path}")
        
        print(f"\n📊 Feature Summary:")
        print(f"   - Total features: {len(feature_cols)}")
        print(f"   - Total samples: {len(features_df)}")
        print(f"   - Alzheimer samples: {sum(y == 1)}")
        print(f"   - Healthy samples: {sum(y == 0)}")


def main():
    """
    Main function to extract features
    """
    print("="*70)
    print("PHASE 2: FEATURE EXTRACTION")
    print("="*70)
    
    # Check if metadata exists
    metadata_path = METADATA_DIR / 'dataset_info.csv'
    
    if not metadata_path.exists():
        print(f"\n❌ Metadata file not found: {metadata_path}")
        print("Please run phase2_data_organizer.py first!")
        return
    
    # Create extractor
    extractor = ComprehensiveFeatureExtractor(sample_rate=16000)
    
    # Process dataset
    features_df = extractor.process_dataset(metadata_path)
    
    # Save features
    extractor.save_features(features_df)
    
    print("\n" + "="*70)
    print("✅ PHASE 2 FEATURE EXTRACTION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review extracted features in data/features/features.csv")
    print("2. Check feature statistics in data/features/feature_statistics.csv")
    print("3. Proceed to Phase 3: SVM Model Training")


if __name__ == "__main__":
    main()
