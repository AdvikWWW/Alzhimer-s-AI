#!/usr/bin/env python3
"""
Run Alzheimer's Detection Model
Interactive script to make predictions on audio files
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import librosa
import soundfile as sf

# Add backend to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from scripts.phase2_feature_extractor import ComprehensiveFeatureExtractor

class AlzheimerDetector:
    """
    Alzheimer's Detection Model Runner
    """
    
    def __init__(self, model_path=None):
        """Initialize the detector with trained model"""
        
        # Find latest model if not specified
        if model_path is None:
            models_dir = PROJECT_ROOT / "models" / "svm"
            model_versions = sorted(models_dir.glob("svm_v_*"), reverse=True)
            
            if not model_versions:
                raise FileNotFoundError("No trained models found in models/svm/")
            
            model_path = model_versions[0]
            print(f"📦 Using latest model: {model_path.name}")
        
        # Load model and scaler
        self.model = joblib.load(model_path / "best_model.joblib")
        self.scaler = joblib.load(model_path / "scaler.joblib")
        
        # Load metadata
        import json
        with open(model_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Training accuracy: {self.metadata.get('best_accuracy', 'N/A')}")
        print(f"   Features: {self.metadata.get('n_features', 'N/A')}")
        
        # Initialize feature extractor
        self.feature_extractor = ComprehensiveFeatureExtractor()
    
    def predict_file(self, audio_path):
        """
        Predict Alzheimer's risk from audio file
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
        
        Returns:
            dict with prediction, probability, and confidence
        """
        audio_path = Path(audio_path)
        
        # Make absolute path
        if not audio_path.is_absolute():
            audio_path = PROJECT_ROOT / audio_path
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"\n🎤 Analyzing: {audio_path.name}")
        
        # Extract features
        print("   Extracting features...")
        features = self.feature_extractor.extract_all_features(audio_path)
        
        if features is None:
            raise ValueError("Failed to extract features from audio file")
        
        # Prepare features for prediction
        # Remove metadata columns
        feature_values = []
        feature_keys = []
        for key in features.keys():
            if key not in ['filename', 'file_path', 'label', 'file_id']:
                feature_values.append(features[key])
                feature_keys.append(key)
        
        X = np.array(feature_values).reshape(1, -1)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Remove zero-variance features (tempo was removed during training)
        # Check if we have more features than expected
        expected_features = self.scaler.n_features_in_
        if X.shape[1] > expected_features:
            # Remove 'tempo' feature if it exists
            if 'tempo' in feature_keys:
                tempo_idx = feature_keys.index('tempo')
                X = np.delete(X, tempo_idx, axis=1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Prepare results
        result = {
            'prediction': 'Alzheimer\'s' if prediction == 1 else 'Healthy',
            'prediction_code': int(prediction),
            'probability_healthy': float(probabilities[0]),
            'probability_alzheimers': float(probabilities[1]),
            'confidence': float(probabilities[prediction]) * 100,
            'features_extracted': len(feature_values)
        }
        
        return result
    
    def print_result(self, result):
        """Print prediction result in a nice format"""
        
        print("\n" + "="*70)
        print("🧠 ALZHEIMER'S DETECTION RESULT")
        print("="*70)
        
        # Main prediction
        if result['prediction'] == 'Alzheimer\'s':
            print(f"\n⚠️  PREDICTION: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
        else:
            print(f"\n✅ PREDICTION: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
        
        # Probabilities
        print(f"\n📊 Detailed Probabilities:")
        print(f"   Healthy:     {result['probability_healthy']*100:.1f}%")
        print(f"   Alzheimer's: {result['probability_alzheimers']*100:.1f}%")
        
        # Features
        print(f"\n🔬 Analysis:")
        print(f"   Features extracted: {result['features_extracted']}")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if result['confidence'] >= 90:
            print(f"   Very high confidence prediction")
        elif result['confidence'] >= 75:
            print(f"   High confidence prediction")
        elif result['confidence'] >= 60:
            print(f"   Moderate confidence prediction")
        else:
            print(f"   Low confidence - borderline case")
        
        print("\n" + "="*70)
        
        # Disclaimer
        print("\n⚠️  DISCLAIMER:")
        print("   This is a research tool and should NOT be used for clinical diagnosis.")
        print("   Always consult healthcare professionals for medical advice.")
        print("="*70 + "\n")


def main():
    """Main function for interactive use"""
    
    print("="*70)
    print("🧠 ALZHEIMER'S VOICE DETECTION SYSTEM")
    print("="*70)
    print("\nModel trained on 50 real recordings")
    print("Accuracy: 90% | Precision: 100%")
    print("="*70 + "\n")
    
    # Initialize detector
    try:
        detector = AlzheimerDetector()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPlease ensure you have trained a model first:")
        print("  python3 backend/scripts/train_svm_simple.py")
        return
    
    # Interactive mode
    if len(sys.argv) > 1:
        # File path provided as argument
        audio_path = sys.argv[1]
        try:
            result = detector.predict_file(audio_path)
            detector.print_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Interactive mode
        print("📁 Available test recordings:")
        
        # List some example files
        data_dir = PROJECT_ROOT / "data" / "processed"
        alz_files = list((data_dir / "alzheimer").glob("*.wav"))[:5]
        healthy_files = list((data_dir / "healthy").glob("*.wav"))[:5]
        
        print("\n   Alzheimer's samples:")
        for i, f in enumerate(alz_files, 1):
            print(f"   {i}. {f.name}")
        
        print("\n   Healthy samples:")
        for i, f in enumerate(healthy_files, 1):
            print(f"   {i+5}. {f.name}")
        
        print("\n" + "="*70)
        print("\nUsage:")
        print("  1. Run with file path: python3 run_model.py <audio_file.wav>")
        print("  2. Or enter file path when prompted below")
        print("\nExample:")
        print("  python3 run_model.py data/processed/alzheimer/alz_001.wav")
        print("="*70 + "\n")
        
        # Prompt for file
        while True:
            audio_path = input("Enter audio file path (or 'q' to quit): ").strip()
            
            if audio_path.lower() in ['q', 'quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            if not audio_path:
                continue
            
            try:
                result = detector.predict_file(audio_path)
                detector.print_result(result)
                
                # Ask if want to analyze another
                another = input("\nAnalyze another file? (y/n): ").strip().lower()
                if another != 'y':
                    print("\n👋 Goodbye!")
                    break
                    
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try another file.\n")


if __name__ == "__main__":
    main()
