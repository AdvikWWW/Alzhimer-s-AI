# 🔧 Scripts Directory - Enhanced Alzheimer's Voice Detection

This directory contains all the enhanced scripts for training, debugging, and analyzing the Alzheimer's voice detection system.

---

## 📁 Files Overview

### 1. **train_model_with_data.py** ⭐
**Purpose:** Complete training pipeline for real audio data

**Features:**
- Loads audio files (Alzheimer1-10.wav, Normal1-10.wav)
- Extracts 100+ features per recording
- Trains ensemble models (RandomForest, XGBoost, GradientBoosting, LogisticRegression)
- Cross-validation and performance metrics
- Saves trained models with versioning
- Generates voice_features.csv

**Usage:**
```bash
python train_model_with_data.py \
    --data-dir /path/to/audio/files \
    --output-dir ./models

# Skip feature extraction if CSV exists
python train_model_with_data.py \
    --data-dir /path/to/audio/files \
    --output-dir ./models \
    --skip-extraction
```

**Input:** Audio files named Alzheimer1.wav, Normal1.wav, etc.

**Output:**
- `models/v{timestamp}/` - Trained models
- `models/voice_features.csv` - Feature matrix
- Console output with performance metrics

---

### 2. **enhanced_word_level_analyzer.py** ⭐
**Purpose:** Advanced word-by-word analysis with deep learning

**Features:**
- Wav2Vec2 speech embeddings
- MFCC deltas (velocity + acceleration)
- Formant dynamics and shifts
- Spectral entropy per word
- Word timing and rhythm analysis
- Intelligent Alzheimer's scoring

**Usage:**
```bash
python enhanced_word_level_analyzer.py \
    --audio /path/to/recording.wav \
    --output word_analysis.json
```

**Key Classes:**
- `WordLevelAnalyzer`: Extracts word-level features
- `IntelligentAlzheimerScorer`: Generates risk scores

**Output:** JSON file with word-level analysis and risk scores

---

### 3. **debug_model_pipeline.py** ⭐
**Purpose:** Comprehensive debugging tool

**Features:**
- Checks feature extraction variation
- Identifies zero-variance features
- Compares Alzheimer vs Normal samples
- Tests model loading and predictions
- Generates debug reports

**Usage:**
```bash
python debug_model_pipeline.py \
    --audio-files Alzheimer1.wav Normal1.wav Alzheimer2.wav Normal2.wav
```

**Output:**
- Console output with detailed analysis
- `debug_features.csv` - Feature comparison

**What It Checks:**
1. ✅ Feature extraction produces variation
2. ✅ Features differ between classes
3. ✅ Models load correctly
4. ✅ Predictions are consistent
5. ✅ Feature vectors are normalized

---

### 4. **streamlit_demo.py** ⭐
**Purpose:** Interactive web demo

**Features:**
- Upload audio files (WAV, MP3, M4A, OGG)
- Real-time processing with progress tracking
- Interactive visualizations
- Risk assessment dashboard
- Downloadable JSON reports

**Usage:**
```bash
streamlit run streamlit_demo.py

# Open browser to http://localhost:8501
```

**Demo Sections:**
1. **Upload Audio**: Analyze pre-recorded files
2. **Record Audio**: Real-time recording (coming soon)
3. **Batch Analysis**: Multiple files (coming soon)

---

### 5. **quick_test.py**
**Purpose:** System verification

**Features:**
- Tests all package imports
- Verifies service initialization
- Checks spaCy model
- Tests GPU availability
- Provides installation guidance

**Usage:**
```bash
python quick_test.py
```

**Output:**
```
🔍 Testing imports...
  ✅ NumPy
  ✅ Pandas
  ...

🔍 Testing services...
  ✅ AudioProcessor initialized
  ...

🎉 ALL TESTS PASSED!
```

---

## 🚀 Quick Start Workflow

### Step 1: Verify Installation
```bash
python quick_test.py
```

### Step 2: Debug Feature Extraction
```bash
python debug_model_pipeline.py \
    --audio-files data/Alzheimer1.wav data/Normal1.wav
```

### Step 3: Train Models
```bash
python train_model_with_data.py \
    --data-dir data/ \
    --output-dir models/
```

### Step 4: Run Demo
```bash
streamlit run streamlit_demo.py
```

---

## 📊 Expected Outputs

### Training Output
```
=== Model Performance Summary ===
random_forest:
  Accuracy: 0.850
  AUC: 0.875
  F1-Score: 0.847
  CV Score: 0.825 ± 0.045

Ensemble Model:
  Accuracy: 0.925
  AUC: 0.925
  F1-Score: 0.922
  CV Score: 0.910 ± 0.032
```

### Debug Output
```
📊 Features with HIGHEST variation:
  acoustic_pause_time_ratio: CV=0.8234
  disfluency_filled_pause_rate: CV=0.7456

📊 Top 10 discriminative features:
  acoustic_pause_time_ratio:
    Alzheimer: 0.3245
    Normal: 0.1523
    Difference: 0.1722 (113.0%)
```

### Analysis Output
```json
{
  "overall_score": 0.72,
  "risk_category": "High_Risk_Possible_Alzheimers",
  "confidence": 0.83,
  "indicators": [
    "Excessive pausing detected",
    "Reduced vocabulary diversity"
  ]
}
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in backend directory:

```bash
# Whisper Configuration
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
WHISPER_DEVICE=cuda  # Options: cuda, cpu

# Audio Processing
SAMPLE_RATE=16000
FRAME_LENGTH=2048
HOP_LENGTH=512
N_MELS=128
N_MFCC=13

# Model Training
ML_MODELS_PATH=./models
```

### Model Selection

Edit in scripts:

```python
# For faster processing (lower accuracy)
WHISPER_MODEL = "base"
use_gpu = False

# For better accuracy (slower)
WHISPER_MODEL = "large-v2"
use_gpu = True
```

---

## 🐛 Troubleshooting

### Issue: "No audio files found"
**Solution:** Check file naming (case-sensitive)
```bash
ls data/
# Should show: Alzheimer1.wav, Alzheimer2.wav, ..., Normal1.wav, Normal2.wav
```

### Issue: "All predictions identical"
**Solution:** Run debug script
```bash
python debug_model_pipeline.py --audio-files data/*.wav
# Check for "Features with ZERO variance"
```

### Issue: "Out of memory"
**Solution:** Use CPU mode or smaller model
```python
# In script, change:
use_gpu = False
WHISPER_MODEL = "base"
```

### Issue: "WhisperX not found"
**Solution:** Install WhisperX
```bash
pip install whisperx
# Or use standard Whisper:
pip install openai-whisper
```

---

## 📈 Performance Tips

### For Faster Processing:
1. Use GPU: `use_gpu=True`
2. Smaller model: `WHISPER_MODEL="base"`
3. Lower sample rate: `SAMPLE_RATE=8000`
4. Skip Wav2Vec2 embeddings

### For Better Accuracy:
1. Larger model: `WHISPER_MODEL="large-v2"`
2. More training data: 50+ samples per class
3. Feature engineering: Add domain-specific features
4. Ensemble models: Use all 4 models

---

## 📚 Additional Resources

- **Complete Guide**: [../ENHANCED_SYSTEM_GUIDE.md](../ENHANCED_SYSTEM_GUIDE.md)
- **Usage Examples**: [../USAGE_EXAMPLES.md](../USAGE_EXAMPLES.md)
- **Implementation Details**: [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)
- **Project Overview**: [../README.md](../README.md)

---

## 🤝 Contributing

To add new scripts:

1. Follow naming convention: `verb_noun.py`
2. Add docstring with purpose and usage
3. Include command-line arguments
4. Add to this README
5. Test with `quick_test.py`

---

## 📄 License

MIT License - See [../LICENSE](../LICENSE) file for details.

---

**Built with ❤️ for Alzheimer's research and early detection**
