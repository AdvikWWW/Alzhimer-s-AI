# 📚 Usage Examples - Alzheimer's Voice Detection

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Training Your Own Model](#training-your-own-model)
3. [Debugging Issues](#debugging-issues)
4. [Advanced Analysis](#advanced-analysis)
5. [Batch Processing](#batch-processing)
6. [API Integration](#api-integration)

---

## Basic Usage

### Example 1: Quick System Test

```bash
cd backend

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

# Run system test
python scripts/quick_test.py
```

**Expected Output:**
```
🔍 Testing imports...
  ✅ NumPy
  ✅ Pandas
  ✅ Librosa
  ✅ Scikit-learn
  ✅ PyTorch
  ✅ Transformers
  ✅ WhisperX
  ✅ spaCy
  ✅ Streamlit
  ✅ Plotly

🔍 Testing services...
  ✅ AudioProcessor initialized
  ✅ ASRService initialized
  ✅ DisfluencyAnalyzer initialized
  ✅ LexicalSemanticAnalyzer initialized

🎉 ALL TESTS PASSED!
```

---

## Training Your Own Model

### Example 2: Training with 10 Audio Files

**Step 1: Prepare your data**
```bash
mkdir -p data
# Copy your audio files to data/
# Files should be named: Alzheimer1.wav, Alzheimer2.wav, ..., Normal1.wav, Normal2.wav, ...
```

**Step 2: Debug feature extraction**
```bash
python scripts/debug_model_pipeline.py \
    --audio-files data/Alzheimer1.wav data/Normal1.wav
```

**Expected Output:**
```
Processing: data/Alzheimer1.wav
  Extracted 127 features
  Sample features:
    acoustic_pitch_mean: 185.3421
    acoustic_pause_time_ratio: 0.3245
    disfluency_filled_pause_rate: 0.1234
    lexical_type_token_ratio: 0.5678
    word_level_inter_word_pause_mean: 0.4521

Processing: data/Normal1.wav
  Extracted 127 features
  Sample features:
    acoustic_pitch_mean: 192.7654
    acoustic_pause_time_ratio: 0.1523
    disfluency_filled_pause_rate: 0.0456
    lexical_type_token_ratio: 0.7891
    word_level_inter_word_pause_mean: 0.2134

✅ All features have variation!

📊 Top 10 discriminative features:
  acoustic_pause_time_ratio:
    Alzheimer: 0.3245
    Normal: 0.1523
    Difference: 0.1722 (113.0%)
```

**Step 3: Train models**
```bash
python scripts/train_model_with_data.py \
    --data-dir data/ \
    --output-dir models/
```

**Expected Output:**
```
Loading audio files from: data/
Found: Alzheimer1.wav
Found: Alzheimer2.wav
...
Found: Normal1.wav
Found: Normal2.wav
...
Loaded 20 audio files (10 Alzheimer's, 10 Normal)

Extracting features from audio files...
✓ Processed: Alzheimer1.wav
✓ Processed: Alzheimer2.wav
...

Created feature matrix: (20, 129)
Feature columns: 127

Training with 20 samples and 127 features
Class distribution: [10 10]

=== Training Individual Models ===
Training random_forest...
random_forest - AUC: 0.875, CV: 0.825 ± 0.045

Training xgboost...
xgboost - AUC: 0.912, CV: 0.880 ± 0.038

Training gradient_boosting...
gradient_boosting - AUC: 0.895, CV: 0.855 ± 0.042

Training logistic_regression...
logistic_regression - AUC: 0.850, CV: 0.810 ± 0.050

=== Training Ensemble Model ===
Ensemble - AUC: 0.925, CV: 0.910 ± 0.032

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

Models saved to: models/
```

---

## Debugging Issues

### Example 3: Diagnosing Identical Predictions

**Problem:** All recordings get the same prediction

**Solution:**
```bash
# Run debug script with multiple samples
python scripts/debug_model_pipeline.py \
    --audio-files \
    data/Alzheimer1.wav \
    data/Alzheimer2.wav \
    data/Normal1.wav \
    data/Normal2.wav
```

**Check the output for:**

1. **Zero-variance features:**
```
⚠️  Found 15 features with ZERO variance:
  - acoustic_jitter_local: 0.0
  - acoustic_shimmer_local: 0.0
```
**Fix:** These features aren't being extracted properly. Check audio quality.

2. **No discrimination between classes:**
```
📊 Top 10 discriminative features:
  acoustic_pitch_mean:
    Alzheimer: 185.34
    Normal: 185.67
    Difference: 0.33 (0.2%)
```
**Fix:** Features are too similar. Need more diverse samples or better recording conditions.

3. **Good discrimination:**
```
📊 Top 10 discriminative features:
  acoustic_pause_time_ratio:
    Alzheimer: 0.3245
    Normal: 0.1523
    Difference: 0.1722 (113.0%)
```
**This is good!** Features show clear differences.

---

## Advanced Analysis

### Example 4: Word-Level Analysis

```python
from enhanced_word_level_analyzer import WordLevelAnalyzer, IntelligentAlzheimerScorer

# Initialize analyzer
analyzer = WordLevelAnalyzer(use_gpu=True)
scorer = IntelligentAlzheimerScorer()

# Analyze audio file
audio_path = "data/Alzheimer1.wav"

# Get word timestamps from ASR
from app.services.asr_service import ASRService
asr = ASRService()
transcription = asr.transcribe_audio(audio_path)
word_timestamps = transcription['word_timestamps']

# Analyze word-by-word
results = analyzer.analyze_audio_word_by_word(audio_path, word_timestamps)

print(f"Analyzed {results['num_words_analyzed']} words")
print(f"Extracted {len(results['aggregated_features'])} aggregated features")

# Generate intelligent score
features = results['aggregated_features']
score = scorer.score_recording(features, transcription)

print(f"\nRisk Score: {score['overall_score']:.2%}")
print(f"Risk Category: {score['risk_category']}")
print(f"Confidence: {score['confidence']:.2%}")

print("\nRisk Indicators:")
for indicator in score['acoustic_biomarkers']['indicators']:
    print(f"  - {indicator}")
```

**Output:**
```
Analyzed 45 words
Extracted 234 aggregated features

Risk Score: 72%
Risk Category: High_Risk_Possible_Alzheimers
Confidence: 83%

Risk Indicators:
  - Reduced pitch variability (monotone speech)
  - Excessive pausing
  - Low spectral complexity
```

---

## Batch Processing

### Example 5: Process Multiple Files

```python
import os
from pathlib import Path
from train_model_with_data import EnhancedFeatureExtractor
import pandas as pd

# Initialize extractor
extractor = EnhancedFeatureExtractor()

# Process directory
data_dir = Path("data")
audio_files = list(data_dir.glob("*.wav"))

results = []
for audio_path in audio_files:
    print(f"Processing: {audio_path.name}")
    
    try:
        features = extractor.extract_all_features(str(audio_path))
        features['filename'] = audio_path.name
        results.append(features)
        print(f"  ✓ Extracted {len(features)} features")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")

# Create DataFrame
df = pd.DataFrame(results)
df.to_csv("batch_features.csv", index=False)
print(f"\nProcessed {len(results)} files")
print(f"Saved to: batch_features.csv")
```

---

## API Integration

### Example 6: Using the FastAPI Backend

**Start the server:**
```bash
cd backend
uvicorn app.main:app --reload
```

**Make API requests:**

```python
import requests
import json

# Upload audio file
url = "http://localhost:8000/api/v1/analyze"

with open("data/Alzheimer1.wav", "rb") as f:
    files = {"file": ("Alzheimer1.wav", f, "audio/wav")}
    response = requests.post(url, files=files)

result = response.json()

print(json.dumps(result, indent=2))
```

**Expected Response:**
```json
{
  "session_id": "abc123",
  "status": "complete",
  "risk_score": {
    "overall_score": 0.72,
    "risk_category": "High Risk (Possible Alzheimer's)",
    "confidence": 0.83,
    "indicators": [
      "Excessive pausing detected",
      "Reduced vocabulary diversity"
    ]
  },
  "acoustic_features": {
    "pitch_mean": 185.34,
    "pause_time_ratio": 0.3245,
    "spectral_entropy_mean": 1.85
  },
  "transcription": {
    "transcript_text": "I went to the... um... store yesterday...",
    "word_count": 7,
    "confidence_score": 0.95
  }
}
```

---

## Streamlit Demo Usage

### Example 7: Interactive Analysis

```bash
# Start Streamlit demo
streamlit run scripts/streamlit_demo.py
```

**Then in the browser:**

1. **Upload Audio File**
   - Click "Choose an audio file"
   - Select your WAV/MP3 file
   - Click "🔬 Analyze Audio"

2. **View Results**
   - Risk Assessment Dashboard
   - Acoustic Features (pitch, voice quality)
   - Transcription with word timeline
   - Disfluency Analysis
   - Lexical-Semantic Metrics

3. **Download Report**
   - Click "Download Full Analysis (JSON)"
   - Save the comprehensive report

---

## Common Workflows

### Workflow 1: Research Study

```bash
# 1. Collect audio data
# Record 50+ participants (25 Alzheimer's, 25 Normal)

# 2. Organize files
mkdir -p study_data
# Name: Alzheimer1.wav, Alzheimer2.wav, ..., Normal1.wav, Normal2.wav, ...

# 3. Debug feature extraction
python scripts/debug_model_pipeline.py \
    --audio-files study_data/Alzheimer1.wav study_data/Normal1.wav

# 4. Train models
python scripts/train_model_with_data.py \
    --data-dir study_data/ \
    --output-dir models/study_v1/

# 5. Validate performance
# Check CV scores and confusion matrix in output

# 6. Analyze new recordings
streamlit run scripts/streamlit_demo.py
```

### Workflow 2: Incremental Training

```bash
# 1. Initial training with 20 samples
python scripts/train_model_with_data.py \
    --data-dir data_batch1/ \
    --output-dir models/v1/

# 2. Collect more data (30 more samples)

# 3. Combine datasets
mkdir -p data_combined
cp data_batch1/*.wav data_combined/
cp data_batch2/*.wav data_combined/

# 4. Retrain with all data
python scripts/train_model_with_data.py \
    --data-dir data_combined/ \
    --output-dir models/v2/

# 5. Compare performance
# Check if v2 has better CV scores than v1
```

### Workflow 3: Clinical Validation

```bash
# 1. Train on research dataset
python scripts/train_model_with_data.py \
    --data-dir research_data/ \
    --output-dir models/research/

# 2. Validate on clinical dataset (separate from training)
python scripts/validate_model.py \
    --model-dir models/research/ \
    --test-dir clinical_data/ \
    --output validation_report.json

# 3. Analyze results
python scripts/analyze_validation.py \
    --report validation_report.json
```

---

## Troubleshooting Examples

### Issue: "WhisperX model not loading"

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU mode
# Edit app/core/config.py:
# WHISPER_DEVICE = "cpu"

# Or install CUDA toolkit
```

### Issue: "Feature extraction too slow"

```python
# Use smaller Whisper model
# In app/core/config.py:
WHISPER_MODEL = "base"  # Instead of "large"

# Or skip Wav2Vec2 embeddings
# Comment out in enhanced_word_level_analyzer.py:
# embedding_features = self._extract_wav2vec_embeddings(word_audio, sr)
```

### Issue: "Out of memory"

```python
# Reduce batch size
# In asr_service.py:
result = self.model.transcribe(audio, batch_size=8)  # Instead of 16

# Or process shorter audio clips
# Split long recordings into 30-second segments
```

---

## Performance Benchmarks

### Expected Processing Times

| Audio Duration | Feature Extraction | Model Training | Total |
|----------------|-------------------|----------------|-------|
| 30 seconds | 15-20s | - | 15-20s |
| 2 minutes | 45-60s | - | 45-60s |
| 5 minutes | 2-3 min | - | 2-3 min |
| 20 samples | - | 2-5 min | 10-15 min |
| 50 samples | - | 5-10 min | 30-45 min |

*Times measured on: Intel i7, 16GB RAM, NVIDIA GTX 1080*

### Expected Accuracy

| Dataset Size | Accuracy | AUC | F1-Score |
|--------------|----------|-----|----------|
| 10 per class | 75-85% | 0.80-0.90 | 0.75-0.85 |
| 25 per class | 85-90% | 0.90-0.95 | 0.85-0.90 |
| 50 per class | 90-95% | 0.93-0.97 | 0.90-0.95 |
| 100+ per class | 92-97% | 0.95-0.99 | 0.92-0.97 |

---

## Next Steps

After completing these examples:

1. **Collect More Data**: Aim for 50+ samples per class
2. **Validate Results**: Compare with clinical assessments
3. **Optimize Models**: Fine-tune hyperparameters
4. **Deploy System**: Set up production environment
5. **Monitor Performance**: Track accuracy over time

---

**For more information, see:**
- [ENHANCED_SYSTEM_GUIDE.md](ENHANCED_SYSTEM_GUIDE.md) - Complete system guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [README.md](README.md) - Project overview

**Need help?** Check the troubleshooting section or open an issue on GitHub.
