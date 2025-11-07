#!/usr/bin/env python3
"""
Phase 2: Data Organizer
Organizes audio files and creates metadata for 150+ recordings
"""

import os
import sys
from pathlib import Path
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import json
from typing import Dict, List
import shutil

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"


class DataOrganizer:
    """
    Organizes audio files and creates metadata
    """
    
    def __init__(self):
        self.metadata = []
        self.stats = {
            'total_files': 0,
            'alzheimer_count': 0,
            'healthy_count': 0,
            'total_duration': 0,
            'avg_duration': 0
        }
    
    def scan_audio_files(self, source_dir: Path) -> List[Dict]:
        """
        Scan a directory for audio files and extract basic info
        """
        audio_files = []
        extensions = ['.wav', '.mp3', '.flac', '.ogg']
        
        for ext in extensions:
            audio_files.extend(source_dir.glob(f'**/*{ext}'))
        
        print(f"Found {len(audio_files)} audio files in {source_dir}")
        
        return audio_files
    
    def analyze_audio_file(self, audio_path: Path) -> Dict:
        """
        Extract basic information from an audio file
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate SNR (simple estimate)
            rms = librosa.feature.rms(y=y)[0]
            signal_power = np.mean(rms ** 2)
            noise_power = np.var(rms)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Quality score (0-10)
            quality_score = min(10, max(0, (snr / 5)))  # Rough estimate
            
            return {
                'duration_sec': float(duration),
                'sample_rate': int(sr),
                'channels': 1 if len(y.shape) == 1 else y.shape[0],
                'samples': len(y),
                'snr_db': float(snr),
                'quality_score': float(quality_score),
                'file_size_mb': audio_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}")
            return None
    
    def create_metadata_entry(self, audio_path: Path, label: str, file_id: str) -> Dict:
        """
        Create a metadata entry for an audio file
        """
        info = self.analyze_audio_file(audio_path)
        
        if info is None:
            return None
        
        entry = {
            'file_id': file_id,
            'filename': audio_path.name,
            'label': label,
            'age': None,  # To be filled manually if available
            'gender': None,  # To be filled manually if available
            'duration_sec': info['duration_sec'],
            'sample_rate': info['sample_rate'],
            'channels': info['channels'],
            'file_size_mb': info['file_size_mb'],
            'snr_db': info['snr_db'],
            'quality_score': info['quality_score'],
            'recording_date': None,  # To be filled manually if available
            'task_type': None,  # To be filled manually if available
            'notes': 'Auto-generated',
            'file_path': str(audio_path.relative_to(PROJECT_ROOT))
        }
        
        return entry
    
    def organize_files(self, source_alzheimer: Path = None, source_healthy: Path = None):
        """
        Organize audio files into proper structure
        """
        print("="*70)
        print("ORGANIZING AUDIO FILES")
        print("="*70)
        
        # Process Alzheimer's files
        if source_alzheimer and source_alzheimer.exists():
            alz_files = self.scan_audio_files(source_alzheimer)
            for i, audio_file in enumerate(alz_files, 1):
                file_id = f"alz_{i:03d}"
                entry = self.create_metadata_entry(audio_file, "Alzheimer", file_id)
                if entry:
                    self.metadata.append(entry)
                    self.stats['alzheimer_count'] += 1
                    self.stats['total_duration'] += entry['duration_sec']
        
        # Process Healthy files
        if source_healthy and source_healthy.exists():
            healthy_files = self.scan_audio_files(source_healthy)
            for i, audio_file in enumerate(healthy_files, 1):
                file_id = f"healthy_{i:03d}"
                entry = self.create_metadata_entry(audio_file, "Healthy", file_id)
                if entry:
                    self.metadata.append(entry)
                    self.stats['healthy_count'] += 1
                    self.stats['total_duration'] += entry['duration_sec']
        
        self.stats['total_files'] = len(self.metadata)
        if self.stats['total_files'] > 0:
            self.stats['avg_duration'] = self.stats['total_duration'] / self.stats['total_files']
        
        print(f"\n✅ Processed {self.stats['total_files']} files")
        print(f"   - Alzheimer: {self.stats['alzheimer_count']}")
        print(f"   - Healthy: {self.stats['healthy_count']}")
        print(f"   - Total duration: {self.stats['total_duration']/60:.1f} minutes")
        print(f"   - Average duration: {self.stats['avg_duration']:.1f} seconds")
    
    def normalize_audio(self, input_path: Path, output_path: Path, target_sr: int = 16000):
        """
        Normalize audio to standard format
        """
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=target_sr, mono=True)
            
            # Normalize amplitude
            y = librosa.util.normalize(y)
            
            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, y, target_sr)
            
            return True
        except Exception as e:
            print(f"Error normalizing {input_path}: {e}")
            return False
    
    def process_and_normalize_all(self):
        """
        Process and normalize all audio files
        """
        print("\n" + "="*70)
        print("NORMALIZING AUDIO FILES")
        print("="*70)
        
        for entry in self.metadata:
            source_path = PROJECT_ROOT / entry['file_path']
            label = entry['label'].lower()
            output_path = PROCESSED_DIR / label / f"{entry['file_id']}.wav"
            
            if self.normalize_audio(source_path, output_path):
                entry['processed_path'] = str(output_path.relative_to(PROJECT_ROOT))
                print(f"✅ Processed: {entry['file_id']}")
            else:
                print(f"❌ Failed: {entry['file_id']}")
    
    def save_metadata(self):
        """
        Save metadata to CSV and JSON
        """
        print("\n" + "="*70)
        print("SAVING METADATA")
        print("="*70)
        
        # Create DataFrame
        df = pd.DataFrame(self.metadata)
        
        # Save CSV
        csv_path = METADATA_DIR / 'dataset_info.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved metadata CSV: {csv_path}")
        
        # Save statistics
        stats_path = METADATA_DIR / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"✅ Saved statistics: {stats_path}")
        
        # Create train/test split (80/20)
        from sklearn.model_selection import train_test_split
        
        indices = list(range(len(df)))
        labels = df['label'].values
        
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )
        
        split_data = {
            'train_indices': train_idx,
            'test_indices': test_idx,
            'train_count': len(train_idx),
            'test_count': len(test_idx)
        }
        
        split_path = METADATA_DIR / 'train_test_split.json'
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"✅ Saved train/test split: {split_path}")
        
        return df
    
    def create_metadata_template(self):
        """
        Create a template CSV for manual data entry
        """
        template_data = {
            'file_id': ['alz_001', 'alz_002', 'healthy_001', 'healthy_002'],
            'filename': ['alz_001.wav', 'alz_002.wav', 'healthy_001.wav', 'healthy_002.wav'],
            'label': ['Alzheimer', 'Alzheimer', 'Healthy', 'Healthy'],
            'age': [72, 68, 70, 65],
            'gender': ['M', 'F', 'M', 'F'],
            'duration_sec': [45.3, 38.7, 42.1, 50.2],
            'recording_date': ['2024-01-15', '2024-01-16', '2024-01-15', '2024-01-17'],
            'task_type': ['verbal_fluency', 'picture_desc', 'verbal_fluency', 'story_recall'],
            'quality_score': [8.5, 7.2, 9.0, 8.8],
            'notes': ['Clear audio', 'Some noise', 'Excellent', 'Good quality']
        }
        
        df = pd.DataFrame(template_data)
        template_path = METADATA_DIR / 'metadata_template.csv'
        df.to_csv(template_path, index=False)
        print(f"✅ Created metadata template: {template_path}")
        
        return template_path
    
    def print_summary(self):
        """
        Print summary of organized data
        """
        print("\n" + "="*70)
        print("DATA ORGANIZATION SUMMARY")
        print("="*70)
        print(f"Total Files: {self.stats['total_files']}")
        print(f"  - Alzheimer: {self.stats['alzheimer_count']}")
        print(f"  - Healthy: {self.stats['healthy_count']}")
        print(f"\nTotal Duration: {self.stats['total_duration']/60:.1f} minutes")
        print(f"Average Duration: {self.stats['avg_duration']:.1f} seconds")
        print(f"\nFiles saved to:")
        print(f"  - Raw: {RAW_AUDIO_DIR}")
        print(f"  - Processed: {PROCESSED_DIR}")
        print(f"  - Metadata: {METADATA_DIR}")
        print("="*70)


def main():
    """
    Main function to organize data
    """
    print("="*70)
    print("PHASE 2: DATA ORGANIZER")
    print("="*70)
    
    organizer = DataOrganizer()
    
    # Option 1: If you already have files in raw_audio folders
    alz_dir = RAW_AUDIO_DIR / "alzheimer"
    healthy_dir = RAW_AUDIO_DIR / "healthy"
    
    if alz_dir.exists() or healthy_dir.exists():
        print("\n📁 Scanning existing audio files...")
        organizer.organize_files(alz_dir, healthy_dir)
        organizer.process_and_normalize_all()
        organizer.save_metadata()
    else:
        print("\n⚠️  No audio files found in data/raw_audio/")
        print("Please add your audio files to:")
        print(f"  - {alz_dir}")
        print(f"  - {healthy_dir}")
        print("\nCreating template for manual entry...")
        organizer.create_metadata_template()
    
    organizer.print_summary()
    
    print("\n✅ Phase 2 Data Organization Complete!")
    print("\nNext steps:")
    print("1. Add your 150+ audio files to data/raw_audio/alzheimer and data/raw_audio/healthy")
    print("2. Run this script again to process them")
    print("3. Proceed to feature extraction (phase2_feature_extractor.py)")


if __name__ == "__main__":
    main()
