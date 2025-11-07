#!/usr/bin/env python3
"""
Create Demo Dataset
Creates a demonstration dataset using existing recordings and synthetic audio
"""

import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import shutil
from scipy import signal

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"


def generate_synthetic_speech(duration=30, sr=16000, label="healthy"):
    """
    Generate synthetic speech-like audio for demonstration
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base frequency (simulating speech)
    if label == "alzheimer":
        # Alzheimer's: more monotone, slower variations
        base_freq = 150 + 20 * np.sin(2 * np.pi * 0.1 * t)
        pause_prob = 0.15  # More pauses
        amplitude_var = 0.3  # Less variation
    else:
        # Healthy: more varied pitch, normal speech
        base_freq = 180 + 40 * np.sin(2 * np.pi * 0.3 * t)
        pause_prob = 0.08  # Fewer pauses
        amplitude_var = 0.6  # More variation
    
    # Generate speech-like signal
    audio = np.zeros_like(t)
    
    for i in range(len(t)):
        if np.random.random() > pause_prob:
            # Add fundamental frequency
            audio[i] += np.sin(2 * np.pi * base_freq[i] * t[i])
            
            # Add harmonics (speech-like)
            audio[i] += 0.5 * np.sin(2 * np.pi * 2 * base_freq[i] * t[i])
            audio[i] += 0.3 * np.sin(2 * np.pi * 3 * base_freq[i] * t[i])
            
            # Add formants (vowel-like resonances)
            audio[i] += 0.4 * np.sin(2 * np.pi * 800 * t[i])
            audio[i] += 0.3 * np.sin(2 * np.pi * 1200 * t[i])
            
            # Amplitude modulation (speech envelope)
            envelope = amplitude_var * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t[i]))
            audio[i] *= envelope
    
    # Add some noise (breath, room tone)
    noise = np.random.normal(0, 0.02, len(audio))
    audio += noise
    
    # Apply bandpass filter (300-3400 Hz, typical telephone quality)
    sos = signal.butter(4, [300, 3400], 'bandpass', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.7
    
    return audio


def create_demo_dataset(num_samples=20):
    """
    Create a demo dataset with synthetic audio
    """
    print("="*70)
    print("CREATING DEMO DATASET")
    print("="*70)
    
    # Create directories
    alz_dir = RAW_AUDIO_DIR / "alzheimer"
    healthy_dir = RAW_AUDIO_DIR / "healthy"
    alz_dir.mkdir(parents=True, exist_ok=True)
    healthy_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing recordings
    existing_recordings = list(RECORDINGS_DIR.glob("*.wav")) if RECORDINGS_DIR.exists() else []
    
    print(f"\nFound {len(existing_recordings)} existing recordings")
    
    # Use existing recordings if available
    used_count = 0
    if existing_recordings:
        print("\nCopying existing recordings...")
        for i, rec in enumerate(existing_recordings[:num_samples//2]):
            # Alternate between alzheimer and healthy
            if i % 2 == 0:
                dest = alz_dir / f"alz_{i+1:03d}.wav"
                label = "Alzheimer"
            else:
                dest = healthy_dir / f"healthy_{i+1:03d}.wav"
                label = "Healthy"
            
            shutil.copy(rec, dest)
            used_count += 1
            print(f"  ✅ Copied to {label}: {dest.name}")
    
    # Generate synthetic audio for remaining samples
    print(f"\nGenerating {num_samples - used_count} synthetic audio samples...")
    
    sr = 16000
    
    # Generate Alzheimer's samples
    alz_needed = (num_samples // 2) - (used_count // 2)
    for i in range(alz_needed):
        idx = (used_count // 2) + i + 1
        audio = generate_synthetic_speech(duration=30, sr=sr, label="alzheimer")
        filename = alz_dir / f"alz_{idx:03d}.wav"
        sf.write(filename, audio, sr)
        print(f"  ✅ Generated Alzheimer's: {filename.name}")
    
    # Generate Healthy samples
    healthy_needed = (num_samples // 2) - (used_count - used_count // 2)
    for i in range(healthy_needed):
        idx = (used_count - used_count // 2) + i + 1
        audio = generate_synthetic_speech(duration=30, sr=sr, label="healthy")
        filename = healthy_dir / f"healthy_{idx:03d}.wav"
        sf.write(filename, audio, sr)
        print(f"  ✅ Generated Healthy: {filename.name}")
    
    # Count final files
    alz_files = list(alz_dir.glob("*.wav"))
    healthy_files = list(healthy_dir.glob("*.wav"))
    
    print("\n" + "="*70)
    print("DEMO DATASET CREATED")
    print("="*70)
    print(f"Alzheimer's files: {len(alz_files)}")
    print(f"Healthy files: {len(healthy_files)}")
    print(f"Total files: {len(alz_files) + len(healthy_files)}")
    print(f"\nFiles location:")
    print(f"  Alzheimer's: {alz_dir}")
    print(f"  Healthy: {healthy_dir}")
    
    return len(alz_files), len(healthy_files)


def main():
    """
    Main function
    """
    print("="*70)
    print("DEMO DATASET CREATOR")
    print("="*70)
    print("\nThis script creates a demonstration dataset for testing.")
    print("It uses existing recordings and generates synthetic audio.")
    print("\nNote: For production, use real Alzheimer's speech data.")
    
    # Create dataset
    alz_count, healthy_count = create_demo_dataset(num_samples=20)
    
    if alz_count > 0 and healthy_count > 0:
        print("\n✅ Demo dataset ready!")
        print("\nNext steps:")
        print("  1. Run: python backend/scripts/phase2_data_organizer.py")
        print("  2. Run: python backend/scripts/phase2_feature_extractor.py")
        print("  3. Run: python backend/scripts/phase2_validate_data.py")
        print("  4. Run: python backend/scripts/svm_model_trainer.py --data-dir data/features")


if __name__ == "__main__":
    main()
