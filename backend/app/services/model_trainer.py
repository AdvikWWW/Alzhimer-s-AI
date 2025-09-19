import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
from datetime import datetime
import json
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    ML model trainer for Alzheimer's voice biomarker detection.
    Supports training on DementiaBank, ADReSS, and custom datasets.
    """
    
    def __init__(self):
        self.models_dir = Path(settings.ML_MODELS_PATH)
        self.models_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'xgboost': {
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'C': 1.0
                }
            }
        }
    
    def prepare_training_data(
        self, 
        acoustic_features: List[Dict[str, Any]],
        lexical_features: List[Dict[str, Any]],
        disfluency_features: List[Dict[str, Any]],
        labels: List[str],
        participant_metadata: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare and combine features for training.
        
        Args:
            acoustic_features: List of acoustic feature dictionaries
            lexical_features: List of lexical-semantic feature dictionaries
            disfluency_features: List of disfluency feature dictionaries
            labels: List of labels ('control', 'mci', 'dementia')
            participant_metadata: List of participant metadata
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        try:
            logger.info("Preparing training data...")
            
            # Extract and combine features
            combined_features = []
            feature_names = []
            
            for i in range(len(acoustic_features)):
                sample_features = []
                
                # Acoustic features
                acoustic = acoustic_features[i]
                for key, value in acoustic.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        sample_features.append(value)
                        if i == 0:  # Only add feature names once
                            feature_names.append(f'acoustic_{key}')
                
                # Lexical features
                lexical = lexical_features[i]
                for key, value in lexical.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        sample_features.append(value)
                        if i == 0:
                            feature_names.append(f'lexical_{key}')
                
                # Disfluency features
                disfluency = disfluency_features[i]
                for key, value in disfluency.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        sample_features.append(value)
                        if i == 0:
                            feature_names.append(f'disfluency_{key}')
                
                # Demographic features
                metadata = participant_metadata[i]
                age = metadata.get('age', 0)
                gender = 1 if metadata.get('gender', '').lower() == 'female' else 0
                education = metadata.get('education_years', 0)
                
                sample_features.extend([age, gender, education])
                if i == 0:
                    feature_names.extend(['age', 'gender', 'education'])
                
                combined_features.append(sample_features)
            
            # Convert to numpy arrays
            X = np.array(combined_features)
            y = np.array(labels)
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y_encoded, feature_names
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def train_individual_models(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train individual models and return performance metrics."""
        try:
            logger.info("Training individual models...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            trained_models = {}
            
            for model_name, config in self.model_configs.items():
                logger.info(f"Training {model_name}...")
                
                # Initialize model
                model = config['class'](**config['params'])
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Calibrate probabilities
                calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                calibrated_model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = calibrated_model.predict(X_test_scaled)
                y_proba = calibrated_model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    calibrated_model, X_train_scaled, y_train, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='roc_auc_ovr'
                )
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(feature_names, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(feature_names, np.abs(model.coef_[0])))
                
                trained_models[model_name] = {
                    'model': calibrated_model,
                    'metrics': metrics,
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': feature_importance
                }
                
                logger.info(f"{model_name} - AUC: {metrics['auc']:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            return trained_models
            
        except Exception as e:
            logger.error(f"Failed to train individual models: {e}")
            raise
    
    def train_ensemble_model(
        self, 
        individual_models: Dict[str, Any], 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Train ensemble model combining individual models."""
        try:
            logger.info("Training ensemble model...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Get predictions from individual models
            train_predictions = []
            test_predictions = []
            
            for model_name, model_data in individual_models.items():
                model = model_data['model']
                train_pred = model.predict_proba(X_train_scaled)
                test_pred = model.predict_proba(X_test_scaled)
                train_predictions.append(train_pred)
                test_predictions.append(test_pred)
            
            # Stack predictions
            X_train_ensemble = np.hstack(train_predictions)
            X_test_ensemble = np.hstack(test_predictions)
            
            # Train meta-learner (logistic regression)
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            meta_learner.fit(X_train_ensemble, y_train)
            
            # Evaluate ensemble
            y_pred = meta_learner.predict(X_test_ensemble)
            y_proba = meta_learner.predict_proba(X_test_ensemble)
            
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            
            # Cross-validation for ensemble
            cv_scores = cross_val_score(
                meta_learner, X_train_ensemble, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc_ovr'
            )
            
            ensemble_model = {
                'meta_learner': meta_learner,
                'individual_models': individual_models,
                'metrics': metrics,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"Ensemble - AUC: {metrics['auc']:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            return ensemble_model
            
        except Exception as e:
            logger.error(f"Failed to train ensemble model: {e}")
            raise
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        try:
            # Basic metrics
            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            
            # AUC score (multiclass)
            if len(np.unique(y_true)) > 2:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            
            return {
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'auc': auc,
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {}
    
    def save_models(
        self, 
        individual_models: Dict[str, Any], 
        ensemble_model: Dict[str, Any],
        feature_names: List[str],
        training_metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """Save trained models and metadata."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_version = f"v{timestamp}"
            
            # Create version directory
            version_dir = self.models_dir / model_version
            version_dir.mkdir(exist_ok=True)
            
            saved_paths = {}
            
            # Save individual models
            for model_name, model_data in individual_models.items():
                model_path = version_dir / f"{model_name}.joblib"
                joblib.dump(model_data['model'], model_path)
                saved_paths[model_name] = str(model_path)
            
            # Save ensemble model
            ensemble_path = version_dir / "ensemble.joblib"
            joblib.dump(ensemble_model['meta_learner'], ensemble_path)
            saved_paths['ensemble'] = str(ensemble_path)
            
            # Save scaler and label encoder
            scaler_path = version_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            saved_paths['scaler'] = str(scaler_path)
            
            encoder_path = version_dir / "label_encoder.joblib"
            joblib.dump(self.label_encoder, encoder_path)
            saved_paths['label_encoder'] = str(encoder_path)
            
            # Save metadata
            metadata = {
                'version': model_version,
                'timestamp': timestamp,
                'feature_names': feature_names,
                'label_classes': self.label_encoder.classes_.tolist(),
                'individual_models': {
                    name: {
                        'metrics': data['metrics'],
                        'cv_scores': data['cv_scores'],
                        'cv_mean': data['cv_mean'],
                        'cv_std': data['cv_std'],
                        'feature_importance': data['feature_importance']
                    }
                    for name, data in individual_models.items()
                },
                'ensemble_metrics': {
                    'metrics': ensemble_model['metrics'],
                    'cv_scores': ensemble_model['cv_scores'],
                    'cv_mean': ensemble_model['cv_mean'],
                    'cv_std': ensemble_model['cv_std']
                },
                'training_metadata': training_metadata
            }
            
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_paths['metadata'] = str(metadata_path)
            
            # Update current model symlink
            current_link = self.models_dir / "current"
            if current_link.exists():
                current_link.unlink()
            current_link.symlink_to(version_dir.name)
            
            logger.info(f"Saved models version {model_version}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise
    
    def load_models(self, version: str = "current") -> Dict[str, Any]:
        """Load trained models and metadata."""
        try:
            model_dir = self.models_dir / version
            if not model_dir.exists():
                raise FileNotFoundError(f"Model version {version} not found")
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load scaler and label encoder
            scaler = joblib.load(model_dir / "scaler.joblib")
            label_encoder = joblib.load(model_dir / "label_encoder.joblib")
            
            # Load individual models
            individual_models = {}
            for model_name in self.model_configs.keys():
                model_path = model_dir / f"{model_name}.joblib"
                if model_path.exists():
                    individual_models[model_name] = joblib.load(model_path)
            
            # Load ensemble model
            ensemble_path = model_dir / "ensemble.joblib"
            ensemble_model = joblib.load(ensemble_path) if ensemble_path.exists() else None
            
            return {
                'individual_models': individual_models,
                'ensemble_model': ensemble_model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def validate_model_performance(
        self, 
        models: Dict[str, Any], 
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """Validate model performance on external validation set."""
        try:
            X_val, y_val = validation_data
            X_val_scaled = models['scaler'].transform(X_val)
            
            results = {}
            
            # Validate individual models
            for model_name, model in models['individual_models'].items():
                y_pred = model.predict(X_val_scaled)
                y_proba = model.predict_proba(X_val_scaled)
                metrics = self._calculate_metrics(y_val, y_pred, y_proba)
                results[model_name] = metrics
            
            # Validate ensemble model
            if models['ensemble_model']:
                # Get predictions from individual models
                predictions = []
                for model in models['individual_models'].values():
                    pred = model.predict_proba(X_val_scaled)
                    predictions.append(pred)
                
                X_val_ensemble = np.hstack(predictions)
                y_pred = models['ensemble_model'].predict(X_val_ensemble)
                y_proba = models['ensemble_model'].predict_proba(X_val_ensemble)
                metrics = self._calculate_metrics(y_val, y_pred, y_proba)
                results['ensemble'] = metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to validate model performance: {e}")
            raise

# Global model trainer instance
model_trainer = ModelTrainer()
