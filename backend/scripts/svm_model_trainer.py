#!/usr/bin/env python3
"""
SVM-Only Model Trainer for Alzheimer's Detection
Uses Support Vector Machines instead of neural networks
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Dict, Tuple
import joblib
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVMAlzheimerTrainer:
    """
    SVM-based trainer for Alzheimer's detection
    Supports multiple kernel types and hyperparameter optimization
    """
    
    def __init__(self, output_dir: str = "models/svm"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
        logger.info(f"SVM Trainer initialized. Output: {self.output_dir}")
    
    def prepare_data(self, features_df: pd.DataFrame, label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training
        """
        # Separate features and labels
        X = features_df.drop(columns=[label_column, 'filename'], errors='ignore')
        y = features_df[label_column].map({'Alzheimer': 1, 'Normal': 0}).values
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features (critical for SVM)
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y
    
    def train_svm_rbf(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray,
                      optimize: bool = False) -> Dict:
        """
        Train SVM with RBF (Radial Basis Function) kernel
        Best for non-linear, complex patterns
        """
        logger.info("Training SVM with RBF kernel...")
        
        if optimize:
            # Grid search for best hyperparameters
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best RBF parameters: {grid_search.best_params_}")
        else:
            model = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
                cache_size=1000
            )
            model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, "SVM-RBF")
        
        self.models['svm_rbf'] = model
        
        if metrics['accuracy'] > self.best_score:
            self.best_score = metrics['accuracy']
            self.best_model = model
            self.best_model_name = 'svm_rbf'
        
        return metrics
    
    def train_svm_linear(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        optimize: bool = False) -> Dict:
        """
        Train SVM with Linear kernel
        Best for linearly separable data, faster training
        """
        logger.info("Training SVM with Linear kernel...")
        
        if optimize:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100]
            }
            
            svm = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best Linear parameters: {grid_search.best_params_}")
        else:
            model = SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42,
                cache_size=1000
            )
            model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, "SVM-Linear")
        
        self.models['svm_linear'] = model
        
        if metrics['accuracy'] > self.best_score:
            self.best_score = metrics['accuracy']
            self.best_model = model
            self.best_model_name = 'svm_linear'
        
        return metrics
    
    def train_svm_poly(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      optimize: bool = False) -> Dict:
        """
        Train SVM with Polynomial kernel
        Best for polynomial relationships in data
        """
        logger.info("Training SVM with Polynomial kernel...")
        
        if optimize:
            param_grid = {
                'C': [0.1, 1, 10],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto']
            }
            
            svm = SVC(kernel='poly', class_weight='balanced', probability=True, random_state=42)
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            logger.info(f"Best Polynomial parameters: {grid_search.best_params_}")
        else:
            model = SVC(
                kernel='poly',
                degree=3,
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
                cache_size=1000
            )
            model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, "SVM-Polynomial")
        
        self.models['svm_poly'] = model
        
        if metrics['accuracy'] > self.best_score:
            self.best_score = metrics['accuracy']
            self.best_model = model
            self.best_model_name = 'svm_poly'
        
        return metrics
    
    def train_svm_sigmoid(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train SVM with Sigmoid kernel
        Similar to neural network activation
        """
        logger.info("Training SVM with Sigmoid kernel...")
        
        model = SVC(
            kernel='sigmoid',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
            cache_size=1000
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, "SVM-Sigmoid")
        
        self.models['svm_sigmoid'] = model
        
        if metrics['accuracy'] > self.best_score:
            self.best_score = metrics['accuracy']
            self.best_model = model
            self.best_model_name = 'svm_sigmoid'
        
        return metrics
    
    def train_all_svms(self, X: np.ndarray, y: np.ndarray, optimize: bool = False) -> Dict:
        """
        Train all SVM variants and return results
        """
        logger.info("Training all SVM models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Train each SVM variant
        results['rbf'] = self.train_svm_rbf(X_train, y_train, X_test, y_test, optimize)
        results['linear'] = self.train_svm_linear(X_train, y_train, X_test, y_test, optimize)
        results['poly'] = self.train_svm_poly(X_train, y_train, X_test, y_test, optimize)
        results['sigmoid'] = self.train_svm_sigmoid(X_train, y_train, X_test, y_test)
        
        logger.info(f"\nBest model: {self.best_model_name} with accuracy: {self.best_score:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict:
        """
        Calculate comprehensive metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"  Confusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def cross_validate_best_model(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation on best model
        """
        if self.best_model is None:
            logger.error("No model trained yet!")
            return {}
        
        logger.info(f"Cross-validating {self.best_model_name}...")
        
        scores = cross_val_score(self.best_model, X, y, cv=cv, scoring='accuracy')
        
        logger.info(f"Cross-validation scores: {scores}")
        logger.info(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            'scores': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std())
        }
    
    def save_models(self) -> Path:
        """
        Save all trained models
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = self.output_dir / f'svm_v_{timestamp}'
        version_dir.mkdir(exist_ok=True)
        
        # Save all SVM models
        for name, model in self.models.items():
            model_path = version_dir / f'{name}.joblib'
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save best model separately
        if self.best_model is not None:
            best_path = version_dir / 'best_model.joblib'
            joblib.dump(self.best_model, best_path)
            logger.info(f"Saved best model ({self.best_model_name}) to {best_path}")
        
        # Save scaler
        scaler_path = version_dir / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'models': list(self.models.keys()),
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'model_type': 'SVM-only (no neural networks)'
        }
        
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"All models saved to {version_dir}")
        
        return version_dir
    
    def predict(self, X: np.ndarray, use_best: bool = True) -> np.ndarray:
        """
        Make predictions using best model or ensemble
        """
        X_scaled = self.scaler.transform(X)
        
        if use_best and self.best_model is not None:
            return self.best_model.predict(X_scaled)
        else:
            # Ensemble voting
            predictions = []
            for model in self.models.values():
                predictions.append(model.predict(X_scaled))
            
            # Majority voting
            predictions = np.array(predictions)
            return np.round(np.mean(predictions, axis=0)).astype(int)
    
    def predict_proba(self, X: np.ndarray, use_best: bool = True) -> np.ndarray:
        """
        Get prediction probabilities
        """
        X_scaled = self.scaler.transform(X)
        
        if use_best and self.best_model is not None:
            return self.best_model.predict_proba(X_scaled)
        else:
            # Ensemble averaging
            probas = []
            for model in self.models.values():
                probas.append(model.predict_proba(X_scaled))
            
            return np.mean(probas, axis=0)


def train_svm_on_audio_data(data_dir: str, output_dir: str = "models/svm", optimize: bool = False):
    """
    Train SVM models on audio data
    """
    from train_model_with_data import EnhancedFeatureExtractor
    
    logger.info("Starting SVM model training (no neural networks)...")
    
    # Extract features
    extractor = EnhancedFeatureExtractor()
    data_path = Path(data_dir)
    
    all_features = []
    
    # Process Alzheimer samples
    for i in range(1, 11):
        audio_file = data_path / f"Alzheimer{i}.wav"
        if audio_file.exists():
            logger.info(f"Processing {audio_file}")
            features = extractor.extract_all_features(str(audio_file))
            features['label'] = 'Alzheimer'
            features['filename'] = audio_file.name
            all_features.append(features)
    
    # Process Normal samples
    for i in range(1, 11):
        audio_file = data_path / f"Normal{i}.wav"
        if audio_file.exists():
            logger.info(f"Processing {audio_file}")
            features = extractor.extract_all_features(str(audio_file))
            features['label'] = 'Normal'
            features['filename'] = audio_file.name
            all_features.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Save features
    features_path = Path(output_dir) / 'svm_features.csv'
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_path, index=False)
    logger.info(f"Saved features to {features_path}")
    
    # Train SVM models
    trainer = SVMAlzheimerTrainer(output_dir)
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Train all SVM variants
    results = trainer.train_all_svms(X, y, optimize=optimize)
    
    # Cross-validate best model
    cv_results = trainer.cross_validate_best_model(X, y, cv=5)
    
    # Save all models
    save_dir = trainer.save_models()
    
    # Print summary
    print("\n" + "="*70)
    print("SVM MODEL TRAINING COMPLETE (NO NEURAL NETWORKS)")
    print("="*70)
    print("\nModel Performance:")
    for kernel, metrics in results.items():
        print(f"\n{kernel.upper()} Kernel:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Best Model: {trainer.best_model_name.upper()}")
    print(f"Best Accuracy: {trainer.best_score:.4f}")
    print(f"Cross-validation Mean: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
    print(f"\nModels saved to: {save_dir}")
    print("="*70)
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SVM Model Training (No Neural Networks)')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with audio files')
    parser.add_argument('--output-dir', type=str, default='models/svm', help='Output directory')
    parser.add_argument('--optimize', action='store_true', help='Use grid search for hyperparameter optimization')
    
    args = parser.parse_args()
    
    trainer = train_svm_on_audio_data(args.data_dir, args.output_dir, args.optimize)
