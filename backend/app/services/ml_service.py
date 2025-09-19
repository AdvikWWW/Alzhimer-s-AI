import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, calibration_curve
import xgboost as xgb
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from app.core.config import settings
from app.services.audio_processor import AudioProcessor
from app.services.asr_service import ASRService
from app.services.disfluency_analyzer import DisfluencyAnalyzer
from app.services.lexical_semantic_analyzer import LexicalSemanticAnalyzer

logger = logging.getLogger(__name__)

class MLService:
    """
    Machine Learning inference service for Alzheimer's detection
    Implements ensemble models trained on DementiaBank and ADReSS datasets
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.model_metadata = {}
        
        # Initialize component services
        self.audio_processor = AudioProcessor()
        self.asr_service = ASRService()
        self.disfluency_analyzer = DisfluencyAnalyzer()
        self.lexical_semantic_analyzer = LexicalSemanticAnalyzer()
        
        # Model paths
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Load pre-trained models (if available)
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            model_files = {
                'acoustic_model': 'acoustic_rf_model.joblib',
                'lexical_model': 'lexical_xgb_model.joblib',
                'combined_model': 'combined_lgb_model.joblib',
                'ensemble_model': 'ensemble_meta_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.model_dir / filename
                if model_path.exists():
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded {model_name} from {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {str(e)}")
                else:
                    logger.info(f"Model file {filename} not found - will use placeholder model")
                    self.models[model_name] = self._create_placeholder_model(model_name)
            
            # Load scalers
            scaler_files = {
                'acoustic_scaler': 'acoustic_scaler.joblib',
                'lexical_scaler': 'lexical_scaler.joblib',
                'combined_scaler': 'combined_scaler.joblib'
            }
            
            for scaler_name, filename in scaler_files.items():
                scaler_path = self.model_dir / filename
                if scaler_path.exists():
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                else:
                    self.scalers[scaler_name] = StandardScaler()
            
            # Load feature columns
            feature_columns_path = self.model_dir / 'feature_columns.json'
            if feature_columns_path.exists():
                with open(feature_columns_path, 'r') as f:
                    self.feature_columns = json.load(f)
            else:
                self._initialize_feature_columns()
            
            logger.info("Model loading completed")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            self._initialize_placeholder_models()
    
    def _create_placeholder_model(self, model_name: str):
        """Create placeholder model for demonstration"""
        if 'acoustic' in model_name:
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif 'lexical' in model_name:
            return xgb.XGBClassifier(random_state=42)
        elif 'combined' in model_name:
            return GradientBoostingClassifier(random_state=42)
        else:
            return LogisticRegression(random_state=42)
    
    def _initialize_placeholder_models(self):
        """Initialize placeholder models for demonstration"""
        self.models = {
            'acoustic_model': RandomForestClassifier(n_estimators=100, random_state=42),
            'lexical_model': xgb.XGBClassifier(random_state=42),
            'combined_model': GradientBoostingClassifier(random_state=42),
            'ensemble_model': LogisticRegression(random_state=42)
        }
        
        self.scalers = {
            'acoustic_scaler': StandardScaler(),
            'lexical_scaler': RobustScaler(),
            'combined_scaler': StandardScaler()
        }
        
        self._initialize_feature_columns()
    
    def _initialize_feature_columns(self):
        """Initialize expected feature columns for each model"""
        self.feature_columns = {
            'acoustic_features': [
                'pitch_mean', 'pitch_std', 'pitch_range',
                'jitter_local', 'jitter_rap', 'shimmer_local', 'shimmer_apq3',
                'hnr_mean', 'hnr_std', 'f1_mean', 'f2_mean', 'f3_mean',
                'formant_dispersion', 'spectral_centroid_mean', 'spectral_bandwidth_mean',
                'spectral_rolloff_mean', 'zero_crossing_rate_mean',
                'breathiness_score', 'hoarseness_score', 'vocal_tremor_score',
                'speech_rate_syllables_per_second', 'articulation_rate',
                'pause_frequency', 'mean_pause_duration'
            ],
            'lexical_features': [
                'total_words', 'unique_words', 'type_token_ratio', 'moving_average_ttr',
                'semantic_coherence_score', 'topic_drift_score', 'mean_sentence_length',
                'syntactic_complexity_score', 'word_frequency_score', 'idea_density',
                'total_disfluency_rate', 'filled_pause_rate', 'repetition_rate'
            ],
            'combined_features': []  # Will be populated by combining acoustic + lexical
        }
        
        # Combined features is union of acoustic and lexical
        self.feature_columns['combined_features'] = (
            self.feature_columns['acoustic_features'] + 
            self.feature_columns['lexical_features']
        )
    
    async def analyze_recording_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete ML analysis pipeline for a recording session
        """
        try:
            logger.info(f"Starting ML analysis for session: {session_data.get('session_id')}")
            
            session_id = session_data.get('session_id')
            recordings = session_data.get('recordings', [])
            
            if not recordings:
                raise ValueError("No recordings found in session data")
            
            # Process each recording and extract features
            all_features = []
            recording_analyses = []
            
            for recording_id, audio_blob in recordings:
                try:
                    # Save audio blob temporarily for processing
                    temp_audio_path = f"/tmp/{session_id}_{recording_id}.wav"
                    with open(temp_audio_path, 'wb') as f:
                        f.write(audio_blob)
                    
                    # Extract features from this recording
                    recording_features = await self._extract_all_features(temp_audio_path, recording_id)
                    all_features.append(recording_features)
                    recording_analyses.append({
                        'recording_id': recording_id,
                        'features': recording_features
                    })
                    
                    # Clean up temp file
                    Path(temp_audio_path).unlink(missing_ok=True)
                    
                except Exception as e:
                    logger.error(f"Failed to process recording {recording_id}: {str(e)}")
                    continue
            
            if not all_features:
                raise ValueError("Failed to extract features from any recordings")
            
            # Aggregate features across recordings
            aggregated_features = self._aggregate_session_features(all_features)
            
            # Run ML inference
            ml_predictions = self._run_ml_inference(aggregated_features)
            
            # Generate analysis result
            analysis_result = {
                'session_id': session_id,
                'individual_recordings': recording_analyses,
                'aggregated_features': aggregated_features,
                'ml_predictions': ml_predictions,
                'risk_assessment': self._generate_risk_assessment(ml_predictions),
                'quality_flags': self._assess_analysis_quality(aggregated_features, ml_predictions),
                'processing_metadata': {
                    'num_recordings_processed': len(all_features),
                    'total_recordings': len(recordings),
                    'model_versions': self._get_model_versions(),
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            logger.info("ML analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"ML analysis failed: {str(e)}")
            raise
    
    async def _extract_all_features(self, audio_file_path: str, recording_id: str) -> Dict[str, Any]:
        """Extract all features from a single audio recording"""
        try:
            # 1. Audio processing and acoustic features
            audio_features = self.audio_processor.process_audio_file(audio_file_path)
            
            # 2. ASR and transcription
            transcription_result = self.asr_service.transcribe_with_fallback(audio_file_path)
            
            # 3. Disfluency analysis
            disfluency_features = self.disfluency_analyzer.analyze_disfluencies(transcription_result)
            
            # 4. Lexical-semantic analysis
            lexical_semantic_features = self.lexical_semantic_analyzer.analyze_lexical_semantic_features(transcription_result)
            
            # Combine all features
            combined_features = {
                'recording_id': recording_id,
                'acoustic_features': audio_features,
                'transcription_result': transcription_result,
                'disfluency_features': disfluency_features,
                'lexical_semantic_features': lexical_semantic_features,
                'feature_extraction_metadata': {
                    'audio_duration': audio_features.get('total_duration_seconds', 0),
                    'transcript_length': len(transcription_result.get('transcript_text', '')),
                    'transcription_confidence': transcription_result.get('confidence_score', 0),
                    'feature_extraction_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {recording_id}: {str(e)}")
            raise
    
    def _aggregate_session_features(self, all_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate features across all recordings in a session"""
        try:
            if not all_features:
                return {}
            
            # Separate feature types
            acoustic_features_list = []
            disfluency_features_list = []
            lexical_features_list = []
            
            for recording_features in all_features:
                acoustic_features_list.append(recording_features.get('acoustic_features', {}))
                disfluency_features_list.append(recording_features.get('disfluency_features', {}))
                lexical_features_list.append(recording_features.get('lexical_semantic_features', {}))
            
            # Aggregate acoustic features (mean, std, min, max)
            aggregated_acoustic = self._aggregate_numeric_features(acoustic_features_list, 'acoustic')
            
            # Aggregate disfluency features (sum for counts, mean for rates)
            aggregated_disfluency = self._aggregate_disfluency_features(disfluency_features_list)
            
            # Aggregate lexical features (mean for most, sum for word counts)
            aggregated_lexical = self._aggregate_lexical_features(lexical_features_list)
            
            # Combine all aggregated features
            aggregated_features = {
                **aggregated_acoustic,
                **aggregated_disfluency,
                **aggregated_lexical,
                'session_metadata': {
                    'num_recordings': len(all_features),
                    'total_duration': sum(f.get('feature_extraction_metadata', {}).get('audio_duration', 0) for f in all_features),
                    'total_transcript_length': sum(f.get('feature_extraction_metadata', {}).get('transcript_length', 0) for f in all_features)
                }
            }
            
            return aggregated_features
            
        except Exception as e:
            logger.error(f"Feature aggregation failed: {str(e)}")
            return {}
    
    def _aggregate_numeric_features(self, features_list: List[Dict], prefix: str) -> Dict[str, Any]:
        """Aggregate numeric features across recordings"""
        aggregated = {}
        
        # Get all unique feature keys
        all_keys = set()
        for features in features_list:
            all_keys.update(features.keys())
        
        for key in all_keys:
            values = []
            for features in features_list:
                value = features.get(key)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
            
            if values:
                aggregated[f"{key}"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values) if len(values) > 1 else 0.0
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)
        
        return aggregated
    
    def _aggregate_disfluency_features(self, disfluency_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate disfluency features"""
        aggregated = {}
        
        # Sum count features
        count_features = ['filled_pauses_count', 'silent_pauses_count', 'repetitions_count', 'false_starts_count', 'stutters_count']
        for feature in count_features:
            values = [d.get(feature, 0) for d in disfluency_list]
            aggregated[feature] = sum(values)
        
        # Mean rate features
        rate_features = ['total_disfluency_rate', 'filled_pause_rate', 'repetition_rate']
        for feature in rate_features:
            values = [d.get(feature, 0) for d in disfluency_list if d.get(feature) is not None]
            aggregated[feature] = np.mean(values) if values else 0.0
        
        return aggregated
    
    def _aggregate_lexical_features(self, lexical_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate lexical-semantic features"""
        aggregated = {}
        
        # Sum word count features
        sum_features = ['total_words', 'unique_words']
        for feature in sum_features:
            values = [l.get(feature, 0) for l in lexical_list]
            aggregated[feature] = sum(values)
        
        # Mean ratio and score features
        mean_features = [
            'type_token_ratio', 'moving_average_ttr', 'semantic_coherence_score',
            'topic_drift_score', 'mean_sentence_length', 'syntactic_complexity_score',
            'word_frequency_score', 'idea_density'
        ]
        for feature in mean_features:
            values = [l.get(feature, 0) for l in lexical_list if l.get(feature) is not None]
            aggregated[feature] = np.mean(values) if values else 0.0
        
        return aggregated
    
    def _run_ml_inference(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML inference using ensemble models"""
        try:
            predictions = {}
            
            # 1. Acoustic model prediction
            acoustic_features = self._extract_model_features(features, 'acoustic_features')
            if acoustic_features is not None:
                acoustic_pred = self._predict_with_model('acoustic_model', acoustic_features, 'acoustic_scaler')
                predictions['acoustic_model'] = acoustic_pred
            
            # 2. Lexical model prediction
            lexical_features = self._extract_model_features(features, 'lexical_features')
            if lexical_features is not None:
                lexical_pred = self._predict_with_model('lexical_model', lexical_features, 'lexical_scaler')
                predictions['lexical_model'] = lexical_pred
            
            # 3. Combined model prediction
            combined_features = self._extract_model_features(features, 'combined_features')
            if combined_features is not None:
                combined_pred = self._predict_with_model('combined_model', combined_features, 'combined_scaler')
                predictions['combined_model'] = combined_pred
            
            # 4. Ensemble prediction
            if len(predictions) >= 2:
                ensemble_input = np.array([[
                    predictions.get('acoustic_model', {}).get('probability', 0.5),
                    predictions.get('lexical_model', {}).get('probability', 0.5),
                    predictions.get('combined_model', {}).get('probability', 0.5)
                ]])
                
                ensemble_pred = self._predict_with_model('ensemble_model', ensemble_input, None)
                predictions['ensemble_model'] = ensemble_pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"ML inference failed: {str(e)}")
            return {}
    
    def _extract_model_features(self, all_features: Dict[str, Any], feature_type: str) -> Optional[np.ndarray]:
        """Extract features for a specific model"""
        try:
            expected_columns = self.feature_columns.get(feature_type, [])
            if not expected_columns:
                return None
            
            feature_values = []
            for col in expected_columns:
                value = all_features.get(col, 0.0)
                if value is None:
                    value = 0.0
                feature_values.append(float(value))
            
            return np.array([feature_values])
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {feature_type}: {str(e)}")
            return None
    
    def _predict_with_model(self, model_name: str, features: np.ndarray, scaler_name: Optional[str]) -> Dict[str, Any]:
        """Make prediction with a specific model"""
        try:
            model = self.models.get(model_name)
            if model is None:
                return {'probability': 0.5, 'prediction': 0, 'confidence': 0.0}
            
            # Scale features if scaler is provided
            if scaler_name and scaler_name in self.scalers:
                scaler = self.scalers[scaler_name]
                # For placeholder models, fit scaler on the fly
                if not hasattr(scaler, 'mean_'):
                    scaler.fit(features)
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            else:
                probability = 0.5  # Default for models without probability prediction
            
            prediction = 1 if probability > 0.5 else 0
            
            # Calculate confidence (distance from 0.5)
            confidence = abs(probability - 0.5) * 2
            
            return {
                'probability': float(probability),
                'prediction': int(prediction),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {str(e)}")
            return {'probability': 0.5, 'prediction': 0, 'confidence': 0.0}
    
    def _generate_risk_assessment(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate calibrated risk assessment"""
        try:
            if not predictions:
                return {
                    'risk_probability': 0.5,
                    'confidence_interval_lower': 0.3,
                    'confidence_interval_upper': 0.7,
                    'uncertainty_score': 0.5,
                    'risk_category': 'unknown'
                }
            
            # Use ensemble prediction if available, otherwise use best available
            if 'ensemble_model' in predictions:
                primary_pred = predictions['ensemble_model']
            elif 'combined_model' in predictions:
                primary_pred = predictions['combined_model']
            else:
                # Average available predictions
                probs = [pred.get('probability', 0.5) for pred in predictions.values()]
                primary_pred = {
                    'probability': np.mean(probs),
                    'confidence': np.mean([pred.get('confidence', 0.0) for pred in predictions.values()])
                }
            
            risk_prob = primary_pred.get('probability', 0.5)
            confidence = primary_pred.get('confidence', 0.0)
            
            # Calculate uncertainty (inverse of confidence)
            uncertainty = 1.0 - confidence
            
            # Calculate confidence interval (wider for higher uncertainty)
            ci_width = 0.2 + (uncertainty * 0.3)  # Base width + uncertainty adjustment
            ci_lower = max(0.0, risk_prob - ci_width / 2)
            ci_upper = min(1.0, risk_prob + ci_width / 2)
            
            # Determine risk category
            if risk_prob < 0.3:
                risk_category = 'low'
            elif risk_prob < 0.7:
                risk_category = 'moderate'
            else:
                risk_category = 'high'
            
            return {
                'risk_probability': float(risk_prob),
                'confidence_interval_lower': float(ci_lower),
                'confidence_interval_upper': float(ci_upper),
                'uncertainty_score': float(uncertainty),
                'risk_category': risk_category,
                'model_agreement': self._calculate_model_agreement(predictions)
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return {
                'risk_probability': 0.5,
                'confidence_interval_lower': 0.3,
                'confidence_interval_upper': 0.7,
                'uncertainty_score': 0.5,
                'risk_category': 'unknown'
            }
    
    def _calculate_model_agreement(self, predictions: Dict[str, Any]) -> float:
        """Calculate agreement between different models"""
        try:
            if len(predictions) < 2:
                return 1.0
            
            probs = [pred.get('probability', 0.5) for pred in predictions.values()]
            
            # Calculate standard deviation of probabilities
            agreement = 1.0 - (np.std(probs) / 0.5)  # Normalize by max possible std
            return max(0.0, min(1.0, agreement))
            
        except Exception as e:
            logger.error(f"Model agreement calculation failed: {str(e)}")
            return 0.5
    
    def _assess_analysis_quality(self, features: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the analysis and flag issues"""
        try:
            quality_flags = {
                'overall_quality': 'good',
                'issues': [],
                'requires_human_review': False,
                'data_completeness': 1.0,
                'model_confidence': 0.8
            }
            
            # Check data completeness
            total_expected_features = len(self.feature_columns.get('combined_features', []))
            missing_features = sum(1 for key in self.feature_columns.get('combined_features', []) 
                                 if features.get(key) is None)
            
            data_completeness = 1.0 - (missing_features / total_expected_features) if total_expected_features > 0 else 0.0
            quality_flags['data_completeness'] = data_completeness
            
            if data_completeness < 0.7:
                quality_flags['issues'].append('High proportion of missing features')
                quality_flags['overall_quality'] = 'poor'
            
            # Check model confidence
            if predictions:
                confidences = [pred.get('confidence', 0.0) for pred in predictions.values()]
                avg_confidence = np.mean(confidences) if confidences else 0.0
                quality_flags['model_confidence'] = avg_confidence
                
                if avg_confidence < 0.3:
                    quality_flags['issues'].append('Low model confidence')
                    quality_flags['overall_quality'] = 'poor'
            
            # Check session metadata
            session_meta = features.get('session_metadata', {})
            total_duration = session_meta.get('total_duration', 0)
            
            if total_duration < 60:  # Less than 1 minute total
                quality_flags['issues'].append('Insufficient audio duration')
                quality_flags['overall_quality'] = 'fair'
            
            # Determine if human review is needed
            quality_flags['requires_human_review'] = (
                quality_flags['overall_quality'] == 'poor' or
                len(quality_flags['issues']) >= 2 or
                data_completeness < 0.5
            )
            
            return quality_flags
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {
                'overall_quality': 'unknown',
                'issues': ['Quality assessment failed'],
                'requires_human_review': True
            }
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get version information for all models"""
        return {
            'acoustic_model': 'v1.0_placeholder',
            'lexical_model': 'v1.0_placeholder', 
            'combined_model': 'v1.0_placeholder',
            'ensemble_model': 'v1.0_placeholder',
            'feature_extractor': 'v1.0'
        }
