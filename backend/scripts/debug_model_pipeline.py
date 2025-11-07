#!/usr/bin/env python3
"""
Debug Script for Model Pipeline
Checks feature extraction, model loading, and prediction consistency
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.audio_processor import AudioProcessor
from app.services.asr_service import ASRService
from app.services.disfluency_analyzer import DisfluencyAnalyzer
from app.services.lexical_semantic_analyzer import LexicalSemanticAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_feature_extraction(audio_files: list):
    """Debug feature extraction to check for variation"""
    
    logger.info("=" * 80)
    logger.info("DEBUGGING FEATURE EXTRACTION")
    logger.info("=" * 80)
    
    audio_processor = AudioProcessor()
    asr_service = ASRService()
    disfluency_analyzer = DisfluencyAnalyzer()
    lexical_analyzer = LexicalSemanticAnalyzer()
    
    all_features = []
    
    for audio_path in audio_files:
        logger.info(f"\nProcessing: {audio_path}")
        
        try:
            # Extract features
            acoustic = audio_processor.process_audio_file(audio_path)
            transcription = asr_service.transcribe_audio(audio_path)
            disfluency = disfluency_analyzer.analyze_disfluencies(transcription)
            lexical = lexical_analyzer.analyze_lexical_semantic_features(transcription)
            
            # Combine features
            features = {
                **{f'acoustic_{k}': v for k, v in acoustic.items() if isinstance(v, (int, float))},
                **{f'disfluency_{k}': v for k, v in disfluency.items() if isinstance(v, (int, float))},
                **{f'lexical_{k}': v for k, v in lexical.items() if isinstance(v, (int, float))}
            }
            
            features['filename'] = Path(audio_path).name
            all_features.append(features)
            
            # Print sample features
            logger.info(f"  Extracted {len(features)} features")
            logger.info(f"  Sample features:")
            for key in list(features.keys())[:5]:
                logger.info(f"    {key}: {features[key]}")
            
        except Exception as e:
            logger.error(f"  Failed: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Analyze feature variation
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE VARIATION ANALYSIS")
    logger.info("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    logger.info(f"\nTotal numeric features: {len(numeric_cols)}")
    
    # Check for zero-variance features
    zero_var_features = []
    for col in numeric_cols:
        if df[col].std() < 1e-10:
            zero_var_features.append(col)
    
    if zero_var_features:
        logger.warning(f"\n⚠️  Found {len(zero_var_features)} features with ZERO variance:")
        for feat in zero_var_features[:10]:
            logger.warning(f"  - {feat}: {df[feat].iloc[0]}")
    else:
        logger.info("\n✅ All features have variation!")
    
    # Show features with highest variation
    logger.info("\n📊 Features with HIGHEST variation:")
    feature_cvs = {}
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if mean != 0:
            cv = std / abs(mean)
            feature_cvs[col] = cv
    
    sorted_features = sorted(feature_cvs.items(), key=lambda x: x[1], reverse=True)
    for feat, cv in sorted_features[:10]:
        logger.info(f"  {feat}: CV={cv:.4f}, mean={df[feat].mean():.4f}, std={df[feat].std():.4f}")
    
    # Compare Alzheimer vs Normal
    logger.info("\n" + "=" * 80)
    logger.info("ALZHEIMER vs NORMAL COMPARISON")
    logger.info("=" * 80)
    
    alzheimer_mask = df['filename'].str.contains('Alzheimer', case=False, na=False)
    normal_mask = df['filename'].str.contains('Normal', case=False, na=False)
    
    if alzheimer_mask.any() and normal_mask.any():
        logger.info("\n📊 Top 10 discriminative features:")
        
        discriminative_features = {}
        for col in numeric_cols:
            alzheimer_mean = df.loc[alzheimer_mask, col].mean()
            normal_mean = df.loc[normal_mask, col].mean()
            
            if normal_mean != 0:
                diff_ratio = abs(alzheimer_mean - normal_mean) / abs(normal_mean)
                discriminative_features[col] = diff_ratio
        
        sorted_disc = sorted(discriminative_features.items(), key=lambda x: x[1], reverse=True)
        for feat, ratio in sorted_disc[:10]:
            alz_val = df.loc[alzheimer_mask, feat].mean()
            norm_val = df.loc[normal_mask, feat].mean()
            logger.info(f"  {feat}:")
            logger.info(f"    Alzheimer: {alz_val:.4f}")
            logger.info(f"    Normal: {norm_val:.4f}")
            logger.info(f"    Difference: {(alz_val - norm_val):.4f} ({ratio*100:.1f}%)")
    
    # Save features
    output_path = Path('debug_features.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Features saved to: {output_path}")
    
    return df


def debug_model_predictions(features_df: pd.DataFrame):
    """Debug model predictions to check consistency"""
    
    logger.info("\n" + "=" * 80)
    logger.info("DEBUGGING MODEL PREDICTIONS")
    logger.info("=" * 80)
    
    # Check if trained models exist
    models_dir = Path('models')
    
    if not models_dir.exists():
        logger.warning("\n⚠️  No models directory found!")
        logger.info("Run training script first: python scripts/train_model_with_data.py --data-dir <path>")
        return
    
    # Look for model files
    model_files = list(models_dir.rglob('*.joblib'))
    
    if not model_files:
        logger.warning("\n⚠️  No trained models found!")
        logger.info("Run training script first: python scripts/train_model_with_data.py --data-dir <path>")
        return
    
    logger.info(f"\n✅ Found {len(model_files)} model files:")
    for mf in model_files:
        logger.info(f"  - {mf}")
    
    # Try to load and predict
    try:
        import joblib
        
        # Find latest model version
        version_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
        if version_dirs:
            latest_version = sorted(version_dirs)[-1]
            logger.info(f"\n📦 Loading models from: {latest_version}")
            
            # Load scaler
            scaler_path = latest_version / 'scaler.joblib'
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                logger.info("  ✅ Scaler loaded")
            
            # Load a model
            model_files_in_version = list(latest_version.glob('*.joblib'))
            for model_file in model_files_in_version:
                if 'scaler' not in model_file.name and 'encoder' not in model_file.name:
                    model = joblib.load(model_file)
                    logger.info(f"  ✅ Model loaded: {model_file.name}")
                    
                    # Try prediction
                    X = features_df.select_dtypes(include=[np.number]).values
                    
                    if hasattr(scaler, 'transform'):
                        X_scaled = scaler.transform(X)
                        logger.info(f"  ✅ Features scaled: {X_scaled.shape}")
                    else:
                        X_scaled = X
                    
                    if hasattr(model, 'predict_proba'):
                        predictions = model.predict_proba(X_scaled)
                        logger.info(f"\n📊 Predictions:")
                        for i, (idx, row) in enumerate(features_df.iterrows()):
                            filename = row.get('filename', f'Sample {i}')
                            prob = predictions[i][1] if len(predictions[i]) > 1 else predictions[i][0]
                            logger.info(f"  {filename}: {prob:.3f} (Class: {'Alzheimer' if prob > 0.5 else 'Normal'})")
                    
                    break
    
    except Exception as e:
        logger.error(f"\n❌ Model loading/prediction failed: {str(e)}")


def main():
    """Main debug script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug model pipeline')
    parser.add_argument('--audio-files', nargs='+', required=True,
                       help='Audio files to test (e.g., Alzheimer1.wav Normal1.wav)')
    
    args = parser.parse_args()
    
    logger.info("🔍 ALZHEIMER'S VOICE DETECTION - DEBUG SCRIPT")
    logger.info("=" * 80)
    
    # Check if files exist
    valid_files = []
    for audio_file in args.audio_files:
        if Path(audio_file).exists():
            valid_files.append(audio_file)
            logger.info(f"✅ Found: {audio_file}")
        else:
            logger.warning(f"❌ Not found: {audio_file}")
    
    if not valid_files:
        logger.error("\n❌ No valid audio files found!")
        return
    
    # Debug feature extraction
    features_df = debug_feature_extraction(valid_files)
    
    # Debug model predictions
    debug_model_predictions(features_df)
    
    logger.info("\n" + "=" * 80)
    logger.info("DEBUG COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
