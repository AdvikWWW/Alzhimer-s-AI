#!/usr/bin/env python3
"""
Simple SVM Trainer
Trains SVM directly from features.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import json
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "svm"


def load_features(features_path):
    """Load features from CSV"""
    print("="*70)
    print("LOADING FEATURES")
    print("="*70)
    
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['label', 'file_id', 'filename', 'file_path']]
    X = df[feature_cols].values
    y = df['label'].map({'Alzheimer': 1, 'Healthy': 0}).values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Alzheimer samples: {sum(y == 1)}")
    print(f"Healthy samples: {sum(y == 0)}")
    
    return X, y, feature_cols


def train_svm(X, y):
    """Train SVM models"""
    print("\n" + "="*70)
    print("TRAINING SVM MODELS")
    print("="*70)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Remove zero variance features
    variances = np.var(X, axis=0)
    non_zero_var = variances > 0
    X = X[:, non_zero_var]
    print(f"Removed {sum(~non_zero_var)} zero-variance features")
    print(f"Remaining features: {X.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    results = {}
    
    # Train SVM-RBF
    print("\n" + "-"*70)
    print("Training SVM with RBF kernel...")
    svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', 
                  probability=True, random_state=42)
    svm_rbf.fit(X_train, y_train)
    
    y_pred = svm_rbf.predict(X_test)
    results['rbf'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    print(f"Accuracy:  {results['rbf']['accuracy']:.4f}")
    print(f"Precision: {results['rbf']['precision']:.4f}")
    print(f"Recall:    {results['rbf']['recall']:.4f}")
    print(f"F1-Score:  {results['rbf']['f1_score']:.4f}")
    
    # Train SVM-Linear
    print("\n" + "-"*70)
    print("Training SVM with Linear kernel...")
    svm_linear = SVC(kernel='linear', C=1.0, class_weight='balanced', 
                     probability=True, random_state=42)
    svm_linear.fit(X_train, y_train)
    
    y_pred = svm_linear.predict(X_test)
    results['linear'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    print(f"Accuracy:  {results['linear']['accuracy']:.4f}")
    print(f"Precision: {results['linear']['precision']:.4f}")
    print(f"Recall:    {results['linear']['recall']:.4f}")
    print(f"F1-Score:  {results['linear']['f1_score']:.4f}")
    
    # Select best model
    best_model_name = 'rbf' if results['rbf']['accuracy'] >= results['linear']['accuracy'] else 'linear'
    best_model = svm_rbf if best_model_name == 'rbf' else svm_linear
    best_accuracy = results[best_model_name]['accuracy']
    
    print("\n" + "="*70)
    print(f"Best Model: SVM-{best_model_name.upper()}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("="*70)
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'svm_rbf': svm_rbf,
        'svm_linear': svm_linear,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'results': results,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std())
    }


def save_models(models_dict):
    """Save trained models"""
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_dir = OUTPUT_DIR / f'svm_v_{timestamp}'
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SVM models
    joblib.dump(models_dict['svm_rbf'], version_dir / 'svm_rbf.joblib')
    joblib.dump(models_dict['svm_linear'], version_dir / 'svm_linear.joblib')
    joblib.dump(models_dict['best_model'], version_dir / 'best_model.joblib')
    joblib.dump(models_dict['scaler'], version_dir / 'scaler.joblib')
    
    print(f"✅ Saved models to: {version_dir}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'best_model': models_dict['best_model_name'],
        'results': models_dict['results'],
        'cv_mean': models_dict['cv_mean'],
        'cv_std': models_dict['cv_std']
    }
    
    with open(version_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata to: {version_dir / 'metadata.json'}")
    
    return version_dir


def main():
    """Main training function"""
    print("="*70)
    print("SVM MODEL TRAINING")
    print("="*70)
    
    # Load features
    X, y, feature_names = load_features(FEATURES_PATH)
    
    # Train models
    models_dict = train_svm(X, y)
    
    # Save models
    save_dir = save_models(models_dict)
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Model: SVM-{models_dict['best_model_name'].upper()}")
    print(f"Test Accuracy: {models_dict['results'][models_dict['best_model_name']]['accuracy']:.4f}")
    print(f"Cross-Validation: {models_dict['cv_mean']:.4f} (+/- {models_dict['cv_std']:.4f})")
    print(f"\nModels saved to: {save_dir}")
    print("\nModel Performance:")
    for model_name, metrics in models_dict['results'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\n" + "="*70)
    print("✅ SVM TRAINING SUCCESSFUL!")
    print("="*70)


if __name__ == "__main__":
    main()
