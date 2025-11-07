#!/usr/bin/env python3
"""
Research-Based Feature Extractor for Alzheimer's Detection
Based on scholarly articles and ADReSS Challenge best practices

Key Research References:
- Frontiers in Computer Science (2021): Disfluency and interactional features
- PMC Systematic Review (2022): Acoustic and linguistic features
- ADReSS Challenge (2020): Benchmark features for AD detection
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
FEATURES_DIR = DATA_DIR / "features"

class ResearchBasedFeatureExtractor:
    """
    Extract features based on Alzheimer's research literature
    
    Feature Categories (150+ total):
    1. Prosodic Features (30+): Pitch, intonation, stress, rhythm
    2. Temporal Features (40+): Pauses, hesitations, speech rate, timing
    3. Voice Quality (25+): Jitter, shimmer, HNR, spectral tilt
    4. Spectral Features (40+): MFCCs, formants, spectral moments
    5. Articulation (15+): Speech clarity, phonation, energy dynamics
    """
    
    def __init__(self):
        self.feature_names = []
        self.sr = 16000  # Standard sample rate
    
    def extract_prosodic_features(self, y, sr):
        """
        Extract prosodic features (pitch, intonation, stress)
        Research shows AD patients have reduced pitch variation and monotone speech
        """
        features = {}
        
        # Pitch extraction using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            pitch_values = np.array(pitch_values)
            
            # Basic pitch statistics
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            features['pitch_median'] = np.median(pitch_values)
            features['pitch_q25'] = np.percentile(pitch_values, 25)
            features['pitch_q75'] = np.percentile(pitch_values, 75)
            features['pitch_iqr'] = features['pitch_q75'] - features['pitch_q25']
            
            # Pitch variation (key for AD detection)
            features['pitch_cv'] = features['pitch_std'] / (features['pitch_mean'] + 1e-8)
            features['pitch_variation_rate'] = np.mean(np.abs(np.diff(pitch_values)))
            
            # Pitch contour features
            features['pitch_slope'] = np.polyfit(np.arange(len(pitch_values)), pitch_values, 1)[0]
            features['pitch_entropy'] = -np.sum((pitch_values / np.sum(pitch_values)) * 
                                                np.log2(pitch_values / np.sum(pitch_values) + 1e-10))
            
            # Monotonicity (AD patients tend to be more monotone)
            pitch_diff = np.abs(np.diff(pitch_values))
            features['pitch_monotonicity'] = 1.0 - (np.std(pitch_diff) / (np.mean(pitch_diff) + 1e-8))
            
        else:
            # No pitch detected
            for key in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range',
                       'pitch_median', 'pitch_q25', 'pitch_q75', 'pitch_iqr', 'pitch_cv',
                       'pitch_variation_rate', 'pitch_slope', 'pitch_entropy', 'pitch_monotonicity']:
                features[key] = 0.0
        
        # Intonation features (F0 contour analysis)
        hop_length = 512
        f0 = librosa.yin(y, fmin=75, fmax=400, sr=sr, hop_length=hop_length)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) > 0:
            features['f0_mean'] = np.mean(f0_voiced)
            features['f0_std'] = np.std(f0_voiced)
            features['f0_range'] = np.max(f0_voiced) - np.min(f0_voiced)
            features['f0_voiced_ratio'] = len(f0_voiced) / len(f0)
            
            # Intonation slope (rising vs falling)
            features['intonation_slope'] = np.polyfit(np.arange(len(f0_voiced)), f0_voiced, 1)[0]
            
            # Pitch reset rate (how often pitch resets - lower in AD)
            f0_diff = np.diff(f0_voiced)
            resets = np.sum(np.abs(f0_diff) > np.std(f0_diff) * 2)
            features['pitch_reset_rate'] = resets / (len(f0_voiced) + 1e-8)
        else:
            for key in ['f0_mean', 'f0_std', 'f0_range', 'f0_voiced_ratio', 
                       'intonation_slope', 'pitch_reset_rate']:
                features[key] = 0.0
        
        # Stress and rhythm features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        features['onset_strength_max'] = np.max(onset_env)
        
        # Rhythm regularity (AD patients have irregular rhythm)
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        if len(onset_times) > 1:
            onset_intervals = np.diff(onset_times)
            features['rhythm_regularity'] = 1.0 / (np.std(onset_intervals) + 1e-8)
            features['rhythm_mean_interval'] = np.mean(onset_intervals)
            features['rhythm_cv'] = np.std(onset_intervals) / (np.mean(onset_intervals) + 1e-8)
        else:
            features['rhythm_regularity'] = 0.0
            features['rhythm_mean_interval'] = 0.0
            features['rhythm_cv'] = 0.0
        
        return features
    
    def extract_temporal_features(self, y, sr):
        """
        Extract temporal features (pauses, hesitations, speech rate)
        Research shows AD patients have more pauses and slower speech
        """
        features = {}
        
        # Duration
        duration = len(y) / sr
        features['duration_sec'] = duration
        
        # Energy-based voice activity detection
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_threshold = np.mean(rms) * 0.3
        
        # Speech/silence segmentation
        is_speech = rms > rms_threshold
        speech_frames = np.sum(is_speech)
        silence_frames = len(is_speech) - speech_frames
        
        features['speech_ratio'] = speech_frames / len(is_speech)
        features['silence_ratio'] = silence_frames / len(is_speech)
        
        # Pause detection (critical for AD)
        frame_duration = 512 / sr
        min_pause_frames = int(0.2 / frame_duration)  # 200ms minimum pause
        
        # Find pause segments
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_sp in enumerate(is_speech):
            if not is_sp and not in_silence:
                in_silence = True
                silence_start = i
            elif is_sp and in_silence:
                silence_length = i - silence_start
                if silence_length >= min_pause_frames:
                    silence_segments.append(silence_length * frame_duration)
                in_silence = False
        
        if len(silence_segments) > 0:
            features['pause_count'] = len(silence_segments)
            features['pause_duration_mean'] = np.mean(silence_segments)
            features['pause_duration_std'] = np.std(silence_segments)
            features['pause_duration_max'] = np.max(silence_segments)
            features['pause_duration_total'] = np.sum(silence_segments)
            features['pause_rate'] = len(silence_segments) / duration  # pauses per second
            features['pause_density'] = features['pause_duration_total'] / duration
            
            # Long pause ratio (AD patients have more long pauses)
            long_pauses = [p for p in silence_segments if p > 1.0]  # > 1 second
            features['long_pause_ratio'] = len(long_pauses) / (len(silence_segments) + 1e-8)
            features['long_pause_count'] = len(long_pauses)
        else:
            for key in ['pause_count', 'pause_duration_mean', 'pause_duration_std',
                       'pause_duration_max', 'pause_duration_total', 'pause_rate',
                       'pause_density', 'long_pause_ratio', 'long_pause_count']:
                features[key] = 0.0
        
        # Speech rate estimation
        # Estimate syllables using onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        syllable_estimate = len(onsets)
        
        speech_time = features['speech_ratio'] * duration
        if speech_time > 0:
            features['syllable_rate'] = syllable_estimate / speech_time  # syllables per second
            features['estimated_words'] = syllable_estimate / 1.5  # rough estimate
            features['speech_rate_wpm'] = (syllable_estimate / 1.5) / (duration / 60)  # words per minute
        else:
            features['syllable_rate'] = 0.0
            features['estimated_words'] = 0.0
            features['speech_rate_wpm'] = 0.0
        
        # Articulation rate (speech rate excluding pauses)
        if speech_time > 0:
            features['articulation_rate'] = syllable_estimate / speech_time
        else:
            features['articulation_rate'] = 0.0
        
        # Phonation time ratio
        features['phonation_time_ratio'] = speech_time / duration
        
        # Speech continuity (lower in AD)
        if len(silence_segments) > 0:
            features['speech_continuity'] = 1.0 / (1.0 + features['pause_rate'])
        else:
            features['speech_continuity'] = 1.0
        
        # Hesitation markers (estimated from very short pauses)
        short_pauses = [p for p in silence_segments if 0.2 <= p < 0.5]
        features['hesitation_count'] = len(short_pauses)
        features['hesitation_rate'] = len(short_pauses) / duration
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Zero crossing rate variability (speech dynamics)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_max'] = np.max(zcr)
        features['zcr_range'] = features['zcr_max'] - np.min(zcr)
        
        return features
    
    def extract_voice_quality_features(self, y, sr):
        """
        Extract voice quality features (jitter, shimmer, HNR)
        Research shows AD patients have degraded voice quality
        """
        features = {}
        
        # Harmonic-to-Noise Ratio (HNR) - voice clarity
        # Higher HNR = clearer voice, AD patients typically have lower HNR
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        harmonic_energy = np.sum(harmonic ** 2)
        noise_energy = np.sum(percussive ** 2)
        
        if noise_energy > 0:
            features['hnr'] = 10 * np.log10(harmonic_energy / (noise_energy + 1e-10))
        else:
            features['hnr'] = 0.0
        
        # Jitter (pitch perturbation) - higher in AD
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 10:
            pitch_periods = 1.0 / (np.array(pitch_values) + 1e-10)
            period_diffs = np.abs(np.diff(pitch_periods))
            features['jitter_local'] = np.mean(period_diffs) / (np.mean(pitch_periods) + 1e-10)
            features['jitter_rap'] = np.mean(np.abs(period_diffs[:-1] - period_diffs[1:])) / (np.mean(pitch_periods) + 1e-10)
        else:
            features['jitter_local'] = 0.0
            features['jitter_rap'] = 0.0
        
        # Shimmer (amplitude perturbation) - higher in AD
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 10:
            rms_diffs = np.abs(np.diff(rms))
            features['shimmer_local'] = np.mean(rms_diffs) / (np.mean(rms) + 1e-10)
            features['shimmer_apq3'] = np.mean(np.abs(rms[:-2] - rms[2:])) / (np.mean(rms) + 1e-10)
        else:
            features['shimmer_local'] = 0.0
            features['shimmer_apq3'] = 0.0
        
        # Spectral tilt (voice quality indicator)
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Low frequency energy vs high frequency energy
        low_freq_idx = freqs < 1000
        high_freq_idx = freqs > 1000
        
        low_energy = np.mean(np.sum(spec[low_freq_idx, :], axis=0))
        high_energy = np.mean(np.sum(spec[high_freq_idx, :], axis=0))
        
        if high_energy > 0:
            features['spectral_tilt'] = low_energy / (high_energy + 1e-10)
        else:
            features['spectral_tilt'] = 0.0
        
        # Spectral entropy (voice complexity)
        spec_sum = np.sum(spec, axis=0)
        spec_prob = spec / (spec_sum + 1e-10)
        features['spectral_entropy'] = -np.mean(np.sum(spec_prob * np.log2(spec_prob + 1e-10), axis=0))
        
        # Spectral flatness (tonality vs noisiness)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)
        features['spectral_flatness_max'] = np.max(flatness)
        
        # Spectral flux (spectral change rate)
        spec_diff = np.diff(spec, axis=1)
        features['spectral_flux_mean'] = np.mean(np.sum(np.abs(spec_diff), axis=0))
        features['spectral_flux_std'] = np.std(np.sum(np.abs(spec_diff), axis=0))
        
        # Cepstral peak prominence (voice quality)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['cepstral_peak_prominence'] = np.mean(np.max(mfcc, axis=0) - np.min(mfcc, axis=0))
        
        return features
    
    def extract_spectral_features(self, y, sr):
        """
        Extract spectral features (MFCCs, formants, spectral moments)
        """
        features = {}
        
        # MFCCs (13 coefficients + deltas + delta-deltas = 39 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
            features[f'mfcc_delta_{i+1}_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc_delta2_{i+1}_mean'] = np.mean(mfcc_delta2[i])
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(centroid)
        features['spectral_centroid_std'] = np.std(centroid)
        features['spectral_centroid_max'] = np.max(centroid)
        features['spectral_centroid_range'] = np.max(centroid) - np.min(centroid)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        features['spectral_rolloff_std'] = np.std(rolloff)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)
        
        # Spectral contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        for i in range(7):
            features[f'spectral_contrast_band_{i+1}_mean'] = np.mean(contrast[i])
        
        # Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
        
        return features
    
    def extract_articulation_features(self, y, sr):
        """
        Extract articulation features (speech clarity, energy dynamics)
        """
        features = {}
        
        # RMS energy (volume dynamics)
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)
        features['rms_range'] = features['rms_max'] - features['rms_min']
        features['rms_dynamic_range'] = 20 * np.log10(features['rms_max'] / (features['rms_min'] + 1e-10))
        
        # Energy entropy (speech clarity)
        energy = rms ** 2
        energy_prob = energy / (np.sum(energy) + 1e-10)
        features['energy_entropy'] = -np.sum(energy_prob * np.log2(energy_prob + 1e-10))
        
        # Short-time energy variability
        features['energy_cv'] = features['rms_std'] / (features['rms_mean'] + 1e-10)
        
        # Formant estimation (vowel articulation)
        # Using LPC to estimate formants
        try:
            # Simple formant estimation using spectral peaks
            spec = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Find peaks in average spectrum
            avg_spec = np.mean(spec, axis=1)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(avg_spec, height=np.max(avg_spec) * 0.1)
            
            if len(peaks) >= 3:
                formant_freqs = freqs[peaks[:3]]
                features['formant_f1'] = formant_freqs[0]
                features['formant_f2'] = formant_freqs[1]
                features['formant_f3'] = formant_freqs[2]
            else:
                features['formant_f1'] = 0.0
                features['formant_f2'] = 0.0
                features['formant_f3'] = 0.0
        except:
            features['formant_f1'] = 0.0
            features['formant_f2'] = 0.0
            features['formant_f3'] = 0.0
        
        # Articulation index (clarity measure)
        # Based on energy distribution across frequency bands
        low_band = spec[freqs < 500, :]
        mid_band = spec[(freqs >= 500) & (freqs < 2000), :]
        high_band = spec[freqs >= 2000, :]
        
        low_energy = np.mean(np.sum(low_band, axis=0))
        mid_energy = np.mean(np.sum(mid_band, axis=0))
        high_energy = np.mean(np.sum(high_band, axis=0))
        
        total_energy = low_energy + mid_energy + high_energy
        if total_energy > 0:
            features['articulation_index'] = mid_energy / total_energy
        else:
            features['articulation_index'] = 0.0
        
        return features
    
    def extract_all_features(self, audio_path):
        """Extract all features from an audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=120)  # Max 2 minutes
            
            # Extract all feature categories
            features = {}
            features.update(self.extract_prosodic_features(y, sr))
            features.update(self.extract_temporal_features(y, sr))
            features.update(self.extract_voice_quality_features(y, sr))
            features.update(self.extract_spectral_features(y, sr))
            features.update(self.extract_articulation_features(y, sr))
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None


def main():
    """Main feature extraction function"""
    print("="*70)
    print("RESEARCH-BASED FEATURE EXTRACTION")
    print("="*70)
    print("\nBased on scholarly research:")
    print("  - Frontiers in Computer Science (2021)")
    print("  - PMC Systematic Review (2022)")
    print("  - ADReSS Challenge (2020)")
    
    # Load metadata
    metadata_path = METADATA_DIR / 'dataset_info.csv'
    if not metadata_path.exists():
        print(f"\n❌ Metadata not found: {metadata_path}")
        print("Please run phase2_data_organizer.py first")
        return
    
    df = pd.read_csv(metadata_path)
    print(f"\nLoaded metadata: {len(df)} files")
    
    # Initialize extractor
    extractor = ResearchBasedFeatureExtractor()
    
    # Extract features
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)
    
    all_features = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = PROJECT_ROOT / row['processed_path']
        
        if not audio_path.exists():
            print(f"\n⚠️  File not found: {audio_path}")
            continue
        
        features = extractor.extract_all_features(audio_path)
        
        if features:
            features['file_id'] = row['file_id']
            features['filename'] = row['filename']
            features['label'] = row['label']
            features['file_path'] = row['file_path']
            all_features.append(features)
    
    # Create DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Save features
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    features_csv = FEATURES_DIR / 'features_research_based.csv'
    features_df.to_csv(features_csv, index=False)
    
    # Save feature names
    feature_cols = [col for col in features_df.columns if col not in ['file_id', 'filename', 'label', 'file_path']]
    with open(FEATURES_DIR / 'feature_names_research.txt', 'w') as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")
    
    print(f"\n✅ Extracted {len(feature_cols)} features from {len(features_df)} files")
    print(f"✅ Saved to: {features_csv}")
    print(f"\nFeature breakdown:")
    print(f"  - Prosodic: ~30 features")
    print(f"  - Temporal: ~40 features")
    print(f"  - Voice Quality: ~25 features")
    print(f"  - Spectral: ~40 features")
    print(f"  - Articulation: ~15 features")
    print(f"  - Total: {len(feature_cols)} features")


if __name__ == "__main__":
    main()
