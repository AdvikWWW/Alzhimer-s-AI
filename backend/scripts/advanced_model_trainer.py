#!/usr/bin/env python3
"""
Advanced Model Trainer for Alzheimer's Detection
Deep learning with attention mechanisms and multi-modal fusion
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepAlzheimerNet(nn.Module):
    """
    Deep neural network with attention mechanism for Alzheimer's detection
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128, 64]):
        super(DeepAlzheimerNet, self).__init__()
        
        # Build deep layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1] // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Binary classification
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classify
        output = self.classifier(attended_features)
        return output


class AlzheimerDataset(Dataset):
    """
    Custom dataset for Alzheimer's detection
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AdvancedModelTrainer:
    """
    Advanced model trainer with multiple algorithms and ensemble methods
    """
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.deep_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
    
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
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y
    
    def train_deep_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """
        Train deep neural network
        """
        logger.info("Training deep neural network...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create datasets
        train_dataset = AlzheimerDataset(X_train, y_train)
        val_dataset = AlzheimerDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_dim = X.shape[1]
        self.deep_model = DeepAlzheimerNet(input_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.deep_model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.deep_model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.deep_model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            train_acc = correct / total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.deep_model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.deep_model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            val_acc = correct / total
            avg_val_loss = val_loss / len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.deep_model.state_dict(), self.output_dir / 'deep_model_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Load best model
        self.deep_model.load_state_dict(torch.load(self.output_dir / 'deep_model_best.pth'))
        
        return val_acc
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray):
        """
        Train ensemble of traditional ML models
        """
        logger.info("Training ensemble models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 1. Random Forest with optimized parameters
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        self.models['random_forest'] = rf
        logger.info(f"Random Forest accuracy: {rf_score:.4f}")
        
        # 2. XGBoost with optimized parameters
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_test, y_test)
        self.models['xgboost'] = xgb_model
        logger.info(f"XGBoost accuracy: {xgb_score:.4f}")
        
        # 3. LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_score = lgb_model.score(X_test, y_test)
        self.models['lightgbm'] = lgb_model
        logger.info(f"LightGBM accuracy: {lgb_score:.4f}")
        
        # 4. Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        gb_score = gb.score(X_test, y_test)
        self.models['gradient_boosting'] = gb
        logger.info(f"Gradient Boosting accuracy: {gb_score:.4f}")
        
        # 5. Support Vector Machine (SVM) with RBF kernel
        logger.info("Training SVM with RBF kernel...")
        svm_rbf = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
            cache_size=1000
        )
        svm_rbf.fit(X_train, y_train)
        svm_rbf_score = svm_rbf.score(X_test, y_test)
        self.models['svm_rbf'] = svm_rbf
        logger.info(f"SVM (RBF) accuracy: {svm_rbf_score:.4f}")
        
        # 6. Support Vector Machine (SVM) with Linear kernel
        logger.info("Training SVM with Linear kernel...")
        svm_linear = SVC(
            kernel='linear',
            C=1.0,
            class_weight='balanced',
            probability=True,
            random_state=42,
            cache_size=1000
        )
        svm_linear.fit(X_train, y_train)
        svm_linear_score = svm_linear.score(X_test, y_test)
        self.models['svm_linear'] = svm_linear
        logger.info(f"SVM (Linear) accuracy: {svm_linear_score:.4f}")
        
        # 7. Support Vector Machine (SVM) with Polynomial kernel
        logger.info("Training SVM with Polynomial kernel...")
        svm_poly = SVC(
            kernel='poly',
            degree=3,
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
            cache_size=1000
        )
        svm_poly.fit(X_train, y_train)
        svm_poly_score = svm_poly.score(X_test, y_test)
        self.models['svm_poly'] = svm_poly
        logger.info(f"SVM (Polynomial) accuracy: {svm_poly_score:.4f}")
        
        return {
            'random_forest': rf_score,
            'xgboost': xgb_score,
            'lightgbm': lgb_score,
            'gradient_boosting': gb_score,
            'svm_rbf': svm_rbf_score,
            'svm_linear': svm_linear_score,
            'svm_poly': svm_poly_score
        }
    
    def create_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """
        Create meta-learner for ensemble prediction
        """
        logger.info("Creating meta-learner...")
        
        # Get predictions from all models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Collect base model predictions
        train_preds = []
        test_preds = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                train_pred = model.predict_proba(X_train)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]
            else:
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            
            train_preds.append(train_pred)
            test_preds.append(test_pred)
        
        # Add deep model predictions if available
        if self.deep_model is not None:
            self.deep_model.eval()
            with torch.no_grad():
                X_train_tensor = torch.FloatTensor(X_train).to(self.device)
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                
                train_outputs = torch.softmax(self.deep_model(X_train_tensor), dim=1)
                test_outputs = torch.softmax(self.deep_model(X_test_tensor), dim=1)
                
                train_preds.append(train_outputs[:, 1].cpu().numpy())
                test_preds.append(test_outputs[:, 1].cpu().numpy())
        
        # Stack predictions
        X_train_meta = np.column_stack(train_preds)
        X_test_meta = np.column_stack(test_preds)
        
        # Train meta-learner
        meta_learner = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        meta_learner.fit(X_train_meta, y_train)
        
        meta_score = meta_learner.score(X_test_meta, y_test)
        logger.info(f"Meta-learner accuracy: {meta_score:.4f}")
        
        self.models['meta_learner'] = meta_learner
        
        return meta_score
    
    def save_models(self):
        """
        Save all trained models
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = self.output_dir / f'v_{timestamp}'
        version_dir.mkdir(exist_ok=True)
        
        # Save traditional models
        for name, model in self.models.items():
            model_path = version_dir / f'{name}.joblib'
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save deep model
        if self.deep_model is not None:
            torch.save({
                'model_state_dict': self.deep_model.state_dict(),
                'model_config': {
                    'input_dim': self.deep_model.feature_extractor[0].in_features,
                    'hidden_dims': [512, 256, 128, 64]
                }
            }, version_dir / 'deep_model.pth')
            logger.info(f"Saved deep model to {version_dir / 'deep_model.pth'}")
        
        # Save scaler
        joblib.dump(self.scaler, version_dir / 'scaler.joblib')
        logger.info(f"Saved scaler to {version_dir / 'scaler.joblib'}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'models': list(self.models.keys()),
            'has_deep_model': self.deep_model is not None,
            'device': str(self.device)
        }
        
        with open(version_dir / 'metadata.json', 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"All models saved to {version_dir}")
        
        return version_dir
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble prediction probabilities
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        
        # Get predictions from all models
        for name, model in self.models.items():
            if name != 'meta_learner':
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_scaled)[:, 1]
                else:
                    pred = model.predict(X_scaled)
                predictions.append(pred)
        
        # Add deep model predictions
        if self.deep_model is not None:
            self.deep_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                outputs = torch.softmax(self.deep_model(X_tensor), dim=1)
                predictions.append(outputs[:, 1].cpu().numpy())
        
        # Use meta-learner if available
        if 'meta_learner' in self.models:
            X_meta = np.column_stack(predictions)
            final_pred = self.models['meta_learner'].predict_proba(X_meta)
            return final_pred
        else:
            # Average ensemble
            return np.mean(predictions, axis=0).reshape(-1, 1)


def train_on_audio_data(data_dir: str, output_dir: str = "models"):
    """
    Train models on audio data (Alzheimer1-10.wav, Normal1-10.wav)
    """
    from train_model_with_data import EnhancedFeatureExtractor
    
    logger.info("Starting advanced model training...")
    
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
    df.to_csv('advanced_features.csv', index=False)
    logger.info(f"Saved features to advanced_features.csv")
    
    # Train models
    trainer = AdvancedModelTrainer(output_dir)
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Train deep model
    deep_accuracy = trainer.train_deep_model(X, y, epochs=200, batch_size=16)
    logger.info(f"Deep model validation accuracy: {deep_accuracy:.4f}")
    
    # Train ensemble models
    ensemble_scores = trainer.train_ensemble_models(X, y)
    
    # Create meta-learner
    meta_score = trainer.create_meta_learner(X, y)
    
    # Save all models
    save_dir = trainer.save_models()
    
    # Print summary
    print("\n" + "="*60)
    print("ADVANCED MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Deep Model Accuracy: {deep_accuracy:.4f}")
    for model, score in ensemble_scores.items():
        print(f"{model}: {score:.4f}")
    print(f"Meta-Learner Accuracy: {meta_score:.4f}")
    print(f"\nModels saved to: {save_dir}")
    print("="*60)
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Model Training')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with audio files')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    trainer = train_on_audio_data(args.data_dir, args.output_dir)
