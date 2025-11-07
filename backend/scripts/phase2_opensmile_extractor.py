#!/usr/bin/env python3
"""
Phase 2: OpenSMILE Feature Extractor (Optional Advanced Features)
Extracts prosodic and voice quality features using OpenSMILE
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict
from tqdm import tqdm

# Try to import opensmile
try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    print("⚠️  OpenSMILE not installed. Install with: pip install opensmile")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
METADATA_DIR = DATA_DIR / "metadata"


class OpenSMILEFeatureExtractor:
    """
    Extract advanced prosodic features using OpenSMILE
    """
    
    def __init__(self):
        if not OPENSMILE_AVAILABLE:
            raise ImportError("OpenSMILE is not installed. Install with: pip install opensmile")
        
        # Initialize OpenSMILE with different feature sets
        self.smile_prosody = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        self.smile_compare = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    
    def extract_gemaps_features(self, audio_path: Path) -> Dict[str, float]:
        """
        Extract GeMAPS features (prosodic and voice quality)
        
        GeMAPS includes:
        - Pitch (F0)
        - Jitter
        - Shimmer
        - Harmonics-to-Noise Ratio (HNR)
        - Formants (F1, F2, F3)
        - Loudness
        - Spectral features
        """
        try:
            features_df = self.smile_prosody.process_file(str(audio_path))
            features = features_df.iloc[0].to_dict()
            
            # Rename features to be more descriptive
            renamed_features = {}
            for key, value in features.items():
                new_key = f"gemaps_{key}"
                renamed_features[new_key] = float(value) if not pd.isna(value) else 0.0
            
            return renamed_features
            
        except Exception as e:
            print(f"Error extracting GeMAPS features from {audio_path}: {e}")
            return {}
    
    def extract_compare_features(self, audio_path: Path) -> Dict[str, float]:
        """
        Extract ComParE features (comprehensive acoustic features)
        
        ComParE includes:
        - Energy and loudness
        - Spectral features
        - MFCCs
        - Voice quality
        - Temporal features
        """
        try:
            features_df = self.smile_compare.process_file(str(audio_path))
            features = features_df.iloc[0].to_dict()
            
            # Rename features
            renamed_features = {}
            for key, value in features.items():
                new_key = f"compare_{key}"
                renamed_features[new_key] = float(value) if not pd.isna(value) else 0.0
            
            return renamed_features
            
        except Exception as e:
            print(f"Error extracting ComParE features from {audio_path}: {e}")
            return {}
    
    def extract_all_opensmile_features(self, audio_path: Path) -> Dict[str, float]:
        """
        Extract all OpenSMILE features
        """
        features = {}
        
        # Extract GeMAPS (prosodic features)
        gemaps = self.extract_gemaps_features(audio_path)
        features.update(gemaps)
        
        # Note: ComParE has 6000+ features, which might be too many
        # Uncomment if you want to use it
        # compare = self.extract_compare_features(audio_path)
        # features.update(compare)
        
        return features
    
    def process_dataset(self, metadata_path: Path) -> pd.DataFrame:
        """
        Process entire dataset with OpenSMILE
        """
        print("="*70)
        print("EXTRACTING OPENSMILE FEATURES")
        print("="*70)
        
        # Load metadata
        metadata_df = pd.read_csv(metadata_path)
        print(f"Loaded metadata: {len(metadata_df)} files")
        
        all_features = []
        
        # Process each file
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="OpenSMILE extraction"):
            # Get processed audio path
            if 'processed_path' in row and pd.notna(row['processed_path']):
                audio_path = PROJECT_ROOT / row['processed_path']
            else:
                audio_path = PROJECT_ROOT / row['file_path']
            
            if not audio_path.exists():
                print(f"⚠️  File not found: {audio_path}")
                continue
            
            # Extract features
            features = self.extract_all_opensmile_features(audio_path)
            
            if features:
                features['label'] = row['label']
                features['file_id'] = row['file_id']
                features['filename'] = audio_path.name
                all_features.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        
        print(f"\n✅ Extracted OpenSMILE features from {len(features_df)} files")
        print(f"   Total OpenSMILE features: {len(features_df.columns) - 3}")
        
        return features_df
    
    def save_features(self, features_df: pd.DataFrame):
        """
        Save OpenSMILE features
        """
        print("\n" + "="*70)
        print("SAVING OPENSMILE FEATURES")
        print("="*70)
        
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = FEATURES_DIR / 'opensmile_features.csv'
        features_df.to_csv(csv_path, index=False)
        print(f"✅ Saved OpenSMILE CSV: {csv_path}")
        
        # Save feature names
        feature_cols = [col for col in features_df.columns if col not in ['label', 'file_id', 'filename']]
        feature_names_path = FEATURES_DIR / 'opensmile_feature_names.txt'
        with open(feature_names_path, 'w') as f:
            for name in feature_cols:
                f.write(f"{name}\n")
        print(f"✅ Saved feature names: {feature_names_path}")
    
    def merge_with_librosa_features(self):
        """
        Merge OpenSMILE features with librosa features
        """
        librosa_path = FEATURES_DIR / 'features.csv'
        opensmile_path = FEATURES_DIR / 'opensmile_features.csv'
        
        if not librosa_path.exists():
            print("⚠️  Librosa features not found. Run phase2_feature_extractor.py first.")
            return
        
        if not opensmile_path.exists():
            print("⚠️  OpenSMILE features not found.")
            return
        
        # Load both
        librosa_df = pd.read_csv(librosa_path)
        opensmile_df = pd.read_csv(opensmile_path)
        
        # Merge on file_id
        merged_df = pd.merge(librosa_df, opensmile_df, on=['file_id', 'label'], suffixes=('_librosa', '_opensmile'))
        
        # Save merged features
        merged_path = FEATURES_DIR / 'features_combined.csv'
        merged_df.to_csv(merged_path, index=False)
        
        print(f"\n✅ Merged features saved: {merged_path}")
        print(f"   Total features: {len(merged_df.columns) - 3}")


def main():
    """
    Main function
    """
    if not OPENSMILE_AVAILABLE:
        print("="*70)
        print("OPENSMILE NOT INSTALLED")
        print("="*70)
        print("\nTo install OpenSMILE:")
        print("  pip install opensmile")
        print("\nOpenSMILE provides advanced prosodic features:")
        print("  - Pitch (F0) with high accuracy")
        print("  - Jitter and Shimmer")
        print("  - Harmonics-to-Noise Ratio (HNR)")
        print("  - Formants (F1, F2, F3, F4)")
        print("  - Voice quality features")
        print("\nNote: OpenSMILE is optional. Librosa features are sufficient for good results.")
        return
    
    print("="*70)
    print("PHASE 2: OPENSMILE FEATURE EXTRACTION (OPTIONAL)")
    print("="*70)
    
    # Check if metadata exists
    metadata_path = METADATA_DIR / 'dataset_info.csv'
    
    if not metadata_path.exists():
        print(f"\n❌ Metadata file not found: {metadata_path}")
        print("Please run phase2_data_organizer.py first!")
        return
    
    # Create extractor
    extractor = OpenSMILEFeatureExtractor()
    
    # Process dataset
    features_df = extractor.process_dataset(metadata_path)
    
    # Save features
    extractor.save_features(features_df)
    
    # Merge with librosa features
    extractor.merge_with_librosa_features()
    
    print("\n" + "="*70)
    print("✅ OPENSMILE FEATURE EXTRACTION COMPLETE!")
    print("="*70)
    print("\nYou can now use either:")
    print("  - features.csv (librosa features only)")
    print("  - opensmile_features.csv (OpenSMILE features only)")
    print("  - features_combined.csv (both combined)")


if __name__ == "__main__":
    main()
