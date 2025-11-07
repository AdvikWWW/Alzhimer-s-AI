#!/usr/bin/env python3
"""
Import Real Recordings from Downloads
Copies real Alzheimer's recordings from Downloads folder and removes synthetic data
"""

import os
import shutil
from pathlib import Path
import librosa
import soundfile as sf

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
DOWNLOADS_DIR = Path("/Users/advikmishra/Downloads")

# Source folders in Downloads
DATASET_FOLDERS = [
    DOWNLOADS_DIR / "dataset",
    DOWNLOADS_DIR / "retraining ",
    DOWNLOADS_DIR / "retrainining_2"
]

def clear_synthetic_data():
    """Remove synthetic demo data"""
    print("="*70)
    print("REMOVING SYNTHETIC DEMO DATA")
    print("="*70)
    
    alz_dir = RAW_AUDIO_DIR / "alzheimer"
    healthy_dir = RAW_AUDIO_DIR / "healthy"
    
    # Remove all files
    if alz_dir.exists():
        for file in alz_dir.glob("*.wav"):
            file.unlink()
            print(f"  ✅ Removed: {file.name}")
    
    if healthy_dir.exists():
        for file in healthy_dir.glob("*.wav"):
            file.unlink()
            print(f"  ✅ Removed: {file.name}")
    
    print("\n✅ Synthetic data removed")


def verify_audio_quality(audio_path):
    """
    Verify audio file is valid and has reasonable quality
    Returns: (is_valid, duration, sample_rate, issues)
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=120)  # Max 2 minutes
        duration = librosa.get_duration(y=y, sr=sr)
        
        issues = []
        
        # Check duration (should be at least 5 seconds)
        if duration < 5:
            issues.append(f"Too short ({duration:.1f}s)")
        
        # Check sample rate (should be at least 8kHz)
        if sr < 8000:
            issues.append(f"Low sample rate ({sr}Hz)")
        
        # Check if audio is too quiet
        rms = librosa.feature.rms(y=y)[0]
        if rms.mean() < 0.001:
            issues.append("Very low volume")
        
        # Check if audio is clipped
        if (abs(y) > 0.99).sum() > len(y) * 0.01:  # More than 1% clipped
            issues.append("Audio clipping detected")
        
        is_valid = len(issues) == 0
        return is_valid, duration, sr, issues
        
    except Exception as e:
        return False, 0, 0, [f"Load error: {str(e)}"]


def import_recordings():
    """Import real recordings from Downloads"""
    print("\n" + "="*70)
    print("IMPORTING REAL RECORDINGS FROM DOWNLOADS")
    print("="*70)
    
    alz_dir = RAW_AUDIO_DIR / "alzheimer"
    healthy_dir = RAW_AUDIO_DIR / "healthy"
    alz_dir.mkdir(parents=True, exist_ok=True)
    healthy_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_found': 0,
        'alzheimer_imported': 0,
        'healthy_imported': 0,
        'skipped': 0,
        'total_duration': 0
    }
    
    skipped_files = []
    
    for folder in DATASET_FOLDERS:
        if not folder.exists():
            print(f"\n⚠️  Folder not found: {folder}")
            continue
        
        print(f"\n📁 Processing: {folder.name}")
        
        # Process Alzheimer's files
        alz_source = folder / "Alzheimers"
        if alz_source.exists():
            alz_files = list(alz_source.glob("*.wav")) + list(alz_source.glob("*.mp3"))
            print(f"  Found {len(alz_files)} Alzheimer's files")
            
            for audio_file in alz_files:
                stats['total_found'] += 1
                
                # Verify quality
                is_valid, duration, sr, issues = verify_audio_quality(audio_file)
                
                if is_valid:
                    # Copy to project
                    dest = alz_dir / f"alz_{stats['alzheimer_imported']+1:03d}.wav"
                    shutil.copy(audio_file, dest)
                    stats['alzheimer_imported'] += 1
                    stats['total_duration'] += duration
                    print(f"    ✅ Imported: {audio_file.name} → {dest.name} ({duration:.1f}s, {sr}Hz)")
                else:
                    stats['skipped'] += 1
                    skipped_files.append((audio_file.name, issues))
                    print(f"    ⚠️  Skipped: {audio_file.name} - {', '.join(issues)}")
        
        # Process Normal/Healthy files
        normal_source = folder / "Normal"
        if normal_source.exists():
            normal_files = list(normal_source.glob("*.wav")) + list(normal_source.glob("*.mp3"))
            print(f"  Found {len(normal_files)} Normal files")
            
            for audio_file in normal_files:
                stats['total_found'] += 1
                
                # Verify quality
                is_valid, duration, sr, issues = verify_audio_quality(audio_file)
                
                if is_valid:
                    # Copy to project
                    dest = healthy_dir / f"healthy_{stats['healthy_imported']+1:03d}.wav"
                    shutil.copy(audio_file, dest)
                    stats['healthy_imported'] += 1
                    stats['total_duration'] += duration
                    print(f"    ✅ Imported: {audio_file.name} → {dest.name} ({duration:.1f}s, {sr}Hz)")
                else:
                    stats['skipped'] += 1
                    skipped_files.append((audio_file.name, issues))
                    print(f"    ⚠️  Skipped: {audio_file.name} - {', '.join(issues)}")
    
    # Summary
    print("\n" + "="*70)
    print("IMPORT SUMMARY")
    print("="*70)
    print(f"Total files found: {stats['total_found']}")
    print(f"Alzheimer's imported: {stats['alzheimer_imported']}")
    print(f"Healthy imported: {stats['healthy_imported']}")
    print(f"Total imported: {stats['alzheimer_imported'] + stats['healthy_imported']}")
    print(f"Skipped (quality issues): {stats['skipped']}")
    print(f"Total duration: {stats['total_duration']/60:.1f} minutes")
    print(f"Average duration: {stats['total_duration']/(stats['alzheimer_imported'] + stats['healthy_imported']):.1f} seconds")
    
    if skipped_files:
        print(f"\n⚠️  Skipped files ({len(skipped_files)}):")
        for filename, issues in skipped_files[:10]:  # Show first 10
            print(f"  - {filename}: {', '.join(issues)}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
    
    return stats


def main():
    """Main import function"""
    print("="*70)
    print("REAL RECORDINGS IMPORT TOOL")
    print("="*70)
    print("\nThis script will:")
    print("  1. Remove synthetic demo data")
    print("  2. Import real recordings from Downloads")
    print("  3. Verify audio quality")
    print("  4. Organize files for training")
    
    # Step 1: Clear synthetic data
    clear_synthetic_data()
    
    # Step 2: Import real recordings
    stats = import_recordings()
    
    # Final message
    print("\n" + "="*70)
    print("✅ IMPORT COMPLETE!")
    print("="*70)
    print(f"\nYou now have {stats['alzheimer_imported'] + stats['healthy_imported']} real recordings:")
    print(f"  - Alzheimer's: {stats['alzheimer_imported']}")
    print(f"  - Healthy: {stats['healthy_imported']}")
    print(f"\nNext steps:")
    print("  1. Run: python3 backend/scripts/phase2_data_organizer.py")
    print("  2. Run: python3 backend/scripts/phase2_feature_extractor_enhanced.py")
    print("  3. Run: python3 backend/scripts/train_svm_simple.py")


if __name__ == "__main__":
    main()
