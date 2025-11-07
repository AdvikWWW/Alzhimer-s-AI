#!/usr/bin/env python3
"""
Download Public Alzheimer's Dataset
Downloads DementiaNet dataset from public sources
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import gdown

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"

# DementiaNet Google Drive links
DEMENTIA_FOLDER = "1GKlvbU57g80-ofCOXGwatDD4U15tpJ4S"
NO_DEMENTIA_FOLDER = "1jm7w7J8SfuwKHpEALIK6uxR9aQZR1q8I"

def download_from_gdrive_folder(folder_id: str, output_dir: Path, label: str):
    """
    Download files from Google Drive folder
    """
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {label.upper()} AUDIO FILES")
    print(f"{'='*70}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use gdown to download folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        print(f"\nDownloading from: {url}")
        print(f"Output directory: {output_dir}")
        
        # Download folder
        gdown.download_folder(url, output=str(output_dir), quiet=False, use_cookies=False)
        
        # Count downloaded files
        audio_files = list(output_dir.glob("*.wav")) + list(output_dir.glob("*.mp3"))
        print(f"\n✅ Downloaded {len(audio_files)} {label} audio files")
        
        return len(audio_files)
        
    except Exception as e:
        print(f"\n❌ Error downloading {label} files: {e}")
        print("\n⚠️  Manual download required:")
        print(f"   1. Visit: https://drive.google.com/drive/folders/{folder_id}")
        print(f"   2. Download all files")
        print(f"   3. Place them in: {output_dir}")
        return 0


def main():
    """
    Main download function
    """
    print("="*70)
    print("DOWNLOADING PUBLIC ALZHEIMER'S DATASET")
    print("="*70)
    print("\nDataset: DementiaNet")
    print("Source: https://github.com/shreyasgite/dementianet")
    print("\nThis dataset contains:")
    print("  - Dementia: Audio from public figures with confirmed dementia")
    print("  - No-Dementia: Audio from healthy individuals (90+ years old)")
    
    # Check if gdown is installed
    try:
        import gdown
    except ImportError:
        print("\n❌ gdown not installed!")
        print("Install with: pip install gdown")
        print("\nAlternatively, download manually:")
        print(f"  Dementia: https://drive.google.com/drive/folders/{DEMENTIA_FOLDER}")
        print(f"  No-Dementia: https://drive.google.com/drive/folders/{NO_DEMENTIA_FOLDER}")
        return
    
    # Download dementia files
    dementia_dir = RAW_AUDIO_DIR / "alzheimer"
    dementia_count = download_from_gdrive_folder(DEMENTIA_FOLDER, dementia_dir, "Dementia")
    
    # Download no-dementia files
    healthy_dir = RAW_AUDIO_DIR / "healthy"
    healthy_count = download_from_gdrive_folder(NO_DEMENTIA_FOLDER, healthy_dir, "No-Dementia")
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Dementia files: {dementia_count}")
    print(f"Healthy files: {healthy_count}")
    print(f"Total files: {dementia_count + healthy_count}")
    
    if dementia_count > 0 and healthy_count > 0:
        print("\n✅ Dataset downloaded successfully!")
        print("\nNext steps:")
        print("  1. Run: python backend/scripts/phase2_data_organizer.py")
        print("  2. Run: python backend/scripts/phase2_feature_extractor.py")
        print("  3. Run: python backend/scripts/phase2_validate_data.py")
    else:
        print("\n⚠️  Download incomplete. Please download manually:")
        print(f"  Dementia: https://drive.google.com/drive/folders/{DEMENTIA_FOLDER}")
        print(f"  No-Dementia: https://drive.google.com/drive/folders/{NO_DEMENTIA_FOLDER}")
        print(f"\nPlace files in:")
        print(f"  {dementia_dir}")
        print(f"  {healthy_dir}")


if __name__ == "__main__":
    main()
