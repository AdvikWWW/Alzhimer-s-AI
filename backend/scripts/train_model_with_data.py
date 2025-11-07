#!/usr/bin/env python3
"""
Enhanced Model Training Script for Alzheimer's Voice Detection
Supports training with real audio files (Alzheimer1.wav - Alzheimer10.wav, Normal1.wav - Normal10.wav)
Implements word-level analysis and advanced feature extraction
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import joblib
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.audio_processor import AudioProcessor
from app.services.asr_service import ASRService
from app.services.disfluency_analyzer import DisfluencyAnalyzer
from app.services.lexical_semantic_analyzer import LexicalSemanticAnalyzer
from app.services.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """Enhanced feature extraction with word-level analysis"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.asr_service = ASRService()
        self.disfluency_analyzer = DisfluencyAnalyzer()
        self.lexical_analyzer = LexicalSemanticAnalyzer()
    
    def extract_all_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract comprehensive features from audio file"""
        try:
            logger.info(f"Processing: {audio_path}")
            
            # 1. Audio processing and acoustic features
            acoustic_features = self.audio_processor.process_audio_file(audio_path)
            
            # 2. Transcription with word-level timestamps
            transcription = self.asr_service.transcribe_audio(audio_path)
            
            # 3. Disfluency analysis
            disfluency_features = self.disfluency_analyzer.analyze_disfluencies(transcription)
            
            # 4. Lexical-semantic analysis
            lexical_features = self.lexical_analyzer.analyze_lexical_semantic_features(transcription)
            
            # 5. Word-level features
            word_level_features = self._extract_word_level_features(transcription)
            
            # Combine all features
            all_features = {
                **self._flatten_dict(acoustic_features, 'acoustic'),
                **self._flatten_dict(disfluency_features, 'disfluency'),
                **self._flatten_dict(lexical_features, 'lexical'),
                **self._flatten_dict(word_level_features, 'word_level')
            }
            
            # Remove non-numeric features
            numeric_features = {k: v for k, v in all_features.items() 
                              if isinstance(v, (int, float, np.number)) and not np.isnan(v)}
            
            logger.info(f"Extracted {len(numeric_features)} features")
            return numeric_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {audio_path}: {str(e)}")
            return {}
    
    def _extract_word_level_features(self, transcription: Dict[str, Any]) -> Dict[str, Any]:
        """Extract word-level timing and rhythm features"""
        try:
            word_timestamps = transcription.get('word_timestamps', [])
            
            if not word_timestamps:
                return {}
            
            # Calculate word durations
            word_durations = []
            inter_word_pauses = []
            
            for i, word_info in enumerate(word_timestamps):
                duration = word_info.get('end', 0) - word_info.get('start', 0)
                word_durations.append(duration)
                
                # Calculate pause to next word
                if i < len(word_timestamps) - 1:
                    next_word = word_timestamps[i + 1]
                    pause = next_word.get('start', 0) - word_info.get('end', 0)
                    if pause > 0:
                        inter_word_pauses.append(pause)
            
            features = {}
            
            if word_durations:
                features['word_duration_mean'] = np.mean(word_durations)
                features['word_duration_std'] = np.std(word_durations)
                features['word_duration_median'] = np.median(word_durations)
                features['word_duration_max'] = np.max(word_durations)
                
            if inter_word_pauses:
                features['inter_word_pause_mean'] = np.mean(inter_word_pauses)
                features['inter_word_pause_std'] = np.std(inter_word_pauses)
                features['inter_word_pause_median'] = np.median(inter_word_pauses)
                features['long_pause_count'] = sum(1 for p in inter_word_pauses if p > 0.5)
                features['hesitation_frequency'] = sum(1 for p in inter_word_pauses if p > 0.3) / len(inter_word_pauses)
            
            # Rhythm features
            if len(word_durations) > 1:
                features['rhythm_variability'] = np.std(word_durations) / (np.mean(word_durations) + 1e-10)
                features['speech_rhythm_score'] = 1.0 / (1.0 + features['rhythm_variability'])
            
            return features
            
        except Exception as e:
            logger.error(f"Word-level feature extraction failed: {str(e)}")
            return {}
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary with prefix"""
        flat = {}
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_dict(value, new_key))
            elif isinstance(value, (list, tuple)):
                continue  # Skip lists
            elif isinstance(value, (int, float, np.number)):
                flat[new_key] = value
        
        return flat


def load_audio_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load audio files from directory
    Expected structure:
    - Alzheimer1.wav to Alzheimer10.wav (label: 1)
    - Normal1.wav to Normal10.wav (label: 0)
    """
    audio_files = []
    labels = []
    
    data_path = Path(data_dir)
    
    # Find Alzheimer files
    for i in range(1, 11):
        alzheimer_file = data_path / f"Alzheimer{i}.wav"
        if alzheimer_file.exists():
            audio_files.append(str(alzheimer_file))
            labels.append(1)  # Alzheimer's
            logger.info(f"Found: {alzheimer_file.name}")
    
    # Find Normal files
    for i in range(1, 11):
        normal_file = data_path / f"Normal{i}.wav"
        if normal_file.exists():
            audio_files.append(str(normal_file))
            labels.append(0)  # Normal
            logger.info(f"Found: {normal_file.name}")
    
    logger.info(f"Loaded {len(audio_files)} audio files ({sum(labels)} Alzheimer's, {len(labels) - sum(labels)} Normal)")
    
    return audio_files, labels


def extract_features_from_dataset(audio_files: List[str], labels: List[int]) -> pd.DataFrame:
    """Extract features from all audio files"""
    extractor = EnhancedFeatureExtractor()
    
    all_features = []
    valid_labels = []
    
    for audio_path, label in zip(audio_files, labels):
        try:
            features = extractor.extract_all_features(audio_path)
            
            if features:
                features['label'] = label
                features['filename'] = Path(audio_path).name
                all_features.append(features)
                valid_labels.append(label)
                logger.info(f"✓ Processed: {Path(audio_path).name}")
            else:
                logger.warning(f"✗ Failed: {Path(audio_path).name}")
                
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Fill missing values with column mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    logger.info(f"Created feature matrix: {df.shape}")
    logger.info(f"Feature columns: {len(df.columns) - 2}")  # Exclude label and filename
    
    return df


def train_models(features_df: pd.DataFrame, output_dir: str):
    """Train ML models on extracted features"""
    
    # Separate features and labels
    X = features_df.drop(['label', 'filename'], axis=1, errors='ignore')
    y = features_df['label'].values
    
    feature_names = X.columns.tolist()
    X = X.values
    
    logger.info(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Print feature statistics for debugging
    logger.info("\n=== Feature Statistics ===")
    for i, name in enumerate(feature_names[:10]):  # Show first 10
        values = X[:, i]
        logger.info(f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, range=[{np.min(values):.4f}, {np.max(values):.4f}]")
    
    # Check for feature variation
    feature_stds = np.std(X, axis=0)
    zero_var_features = np.sum(feature_stds < 1e-10)
    if zero_var_features > 0:
        logger.warning(f"Found {zero_var_features} features with near-zero variance")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Create dummy metadata (since we don't have separate feature dicts)
    acoustic_features = [{}] * len(y)
    lexical_features = [{}] * len(y)
    disfluency_features = [{}] * len(y)
    participant_metadata = [{'age': 70, 'gender': 'unknown', 'education_years': 12}] * len(y)
    
    # Train individual models
    logger.info("\n=== Training Individual Models ===")
    trained_models = trainer.train_individual_models(X, y, feature_names)
    
    # Train ensemble model
    logger.info("\n=== Training Ensemble Model ===")
    ensemble_model = trainer.train_ensemble_model(trained_models, X, y)
    
    # Save models
    logger.info("\n=== Saving Models ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_metadata = {
        'num_samples': len(y),
        'num_features': X.shape[1],
        'class_distribution': np.bincount(y).tolist(),
        'training_date': datetime.now().isoformat()
    }
    
    saved_paths = trainer.save_models(
        trained_models,
        ensemble_model,
        feature_names,
        training_metadata
    )
    
    # Save feature DataFrame for reference
    features_df.to_csv(output_path / 'voice_features.csv', index=False)
    logger.info(f"Saved features to: {output_path / 'voice_features.csv'}")
    
    # Print model performance summary
    logger.info("\n=== Model Performance Summary ===")
    for model_name, model_data in trained_models.items():
        metrics = model_data['metrics']
        logger.info(f"{model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"  AUC: {metrics['auc']:.3f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
        logger.info(f"  CV Score: {model_data['cv_mean']:.3f} ± {model_data['cv_std']:.3f}")
    
    logger.info(f"\nEnsemble Model:")
    logger.info(f"  Accuracy: {ensemble_model['metrics']['accuracy']:.3f}")
    logger.info(f"  AUC: {ensemble_model['metrics']['auc']:.3f}")
    logger.info(f"  F1-Score: {ensemble_model['metrics']['f1_score']:.3f}")
    logger.info(f"  CV Score: {ensemble_model['cv_mean']:.3f} ± {ensemble_model['cv_std']:.3f}")
    
    return saved_paths


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Alzheimer\'s voice detection model')
    parser.add_argument('--data-dir', type=str, required=True, 
                       help='Directory containing audio files (Alzheimer1.wav, Normal1.wav, etc.)')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for trained models')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip feature extraction and use existing voice_features.csv')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ALZHEIMER'S VOICE DETECTION - MODEL TRAINING")
    logger.info("=" * 80)
    
    # Load or extract features
    features_csv = Path(args.output_dir) / 'voice_features.csv'
    
    if args.skip_extraction and features_csv.exists():
        logger.info(f"\nLoading existing features from: {features_csv}")
        features_df = pd.read_csv(features_csv)
    else:
        # Load audio dataset
        logger.info(f"\nLoading audio files from: {args.data_dir}")
        audio_files, labels = load_audio_dataset(args.data_dir)
        
        if len(audio_files) == 0:
            logger.error("No audio files found! Please check the data directory.")
            return
        
        # Extract features
        logger.info("\nExtracting features from audio files...")
        features_df = extract_features_from_dataset(audio_files, labels)
        
        if len(features_df) == 0:
            logger.error("Feature extraction failed for all files!")
            return
    
    # Train models
    logger.info("\nTraining models...")
    saved_paths = train_models(features_df, args.output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nModels saved to: {args.output_dir}")
    logger.info("\nTo use the trained models, update ml_service.py to load from:")
    for name, path in saved_paths.items():
        logger.info(f"  {name}: {path}")


if __name__ == "__main__":
    main()
