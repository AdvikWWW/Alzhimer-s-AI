# 🚀 Phase 2 Quick Start Guide

## Complete Step-by-Step Instructions

---

## 📋 Prerequisites

```bash
# Install required packages
pip install librosa soundfile numpy pandas scikit-learn matplotlib seaborn tqdm

# Optional: Install OpenSMILE for advanced features
pip install opensmile
```

---

## 🎯 Step 1: Organize Your Audio Files

### Option A: You Have 150+ Audio Files

1. **Place your files in the correct folders:**

```bash
data/raw_audio/alzheimer/     # Put Alzheimer's patient recordings here
data/raw_audio/healthy/       # Put healthy control recordings here
```

2. **File naming convention (recommended):**
```
Alzheimer's: alz_001.wav, alz_002.wav, ..., alz_075.wav
Healthy: healthy_001.wav, healthy_002.wav, ..., healthy_075.wav
```

3. **Run the organizer:**
```bash
python backend/scripts/phase2_data_organizer.py
```

**Output:**
- Normalized audio files in `data/processed/`
- Metadata in `data/metadata/dataset_info.csv`
- Train/test split in `data/metadata/train_test_split.json`

---

### Option B: You Don't Have Files Yet (Testing)

Use the existing sample files:

```bash
# The system already has some sample files
# Run the organizer to process them
python backend/scripts/phase2_data_organizer.py
```

---

## 🎯 Step 2: Extract Features

### Basic Feature Extraction (Librosa)

```bash
python backend/scripts/phase2_feature_extractor.py
```

**This extracts ~150 features:**
- 39 MFCCs (13 + deltas + delta-deltas)
- Spectral features (centroid, rolloff, bandwidth, contrast, flatness)
- Temporal features (RMS energy, tempo, onset strength)
- Pitch features (mean, std, range, variation)
- Voice quality (HNR, jitter, shimmer estimates)
- Speech timing (pause count, duration, speech rate, pause density)

**Output:**
- `data/features/features.csv` - Main feature file
- `data/features/features.npy` - NumPy format
- `data/features/labels.npy` - Labels
- `data/features/feature_names.txt` - Feature names
- `data/features/feature_statistics.csv` - Statistics

**Time:** ~30-60 minutes for 150 files

---

### Advanced Feature Extraction (OpenSMILE) - Optional

```bash
python backend/scripts/phase2_opensmile_extractor.py
```

**This adds 80+ advanced features:**
- Accurate pitch (F0) tracking
- Jitter and shimmer
- Harmonics-to-Noise Ratio (HNR)
- Formants (F1, F2, F3, F4)
- Voice quality metrics

**Output:**
- `data/features/opensmile_features.csv`
- `data/features/features_combined.csv` - All features merged

**Time:** ~45-90 minutes for 150 files

---

## 🎯 Step 3: Validate Dataset

```bash
python backend/scripts/phase2_validate_data.py
```

**This checks:**
- Metadata completeness
- Feature quality
- Missing values
- Infinite values
- Zero variance features
- Class imbalance
- Outliers

**Output:**
- Validation report
- Visualizations in `data/features/visualizations/`
  - `label_distribution.png`
  - `feature_correlation.png`
  - `feature_distributions.png`

---

## 📊 Expected Results

### After Step 1 (Data Organization)

```
✅ Processed 150 files
   - Alzheimer: 75
   - Healthy: 75
   - Total duration: 112.5 minutes
   - Average duration: 45.0 seconds
```

### After Step 2 (Feature Extraction)

```
✅ Extracted features from 150 files
   Total features per file: 147

📊 Feature Summary:
   - Total features: 147
   - Total samples: 150
   - Alzheimer samples: 75
   - Healthy samples: 75
```

### After Step 3 (Validation)

```
✅ No critical issues found!

⚠️  Found 2 warnings:
   - Found 3 features with zero variance
   - Found 12 features with outliers

📊 Dataset Statistics:
   total_samples: 150
   label_distribution:
      Alzheimer: 75
      Healthy: 75
   total_features: 147
   zero_variance_features: 3

✅ DATASET IS READY FOR TRAINING!
```

---

## 🗂️ Final Folder Structure

```
alzheimer-voice-detection/
├── data/
│   ├── raw_audio/
│   │   ├── alzheimer/           # 75 original files
│   │   └── healthy/             # 75 original files
│   │
│   ├── processed/
│   │   ├── alzheimer/           # 75 normalized files
│   │   └── healthy/             # 75 normalized files
│   │
│   ├── features/
│   │   ├── features.csv         # Main feature dataset ✅
│   │   ├── features.npy         # NumPy format
│   │   ├── labels.npy           # Labels
│   │   ├── feature_names.txt    # Feature list
│   │   ├── feature_statistics.csv
│   │   ├── opensmile_features.csv (optional)
│   │   ├── features_combined.csv (optional)
│   │   └── visualizations/
│   │       ├── label_distribution.png
│   │       ├── feature_correlation.png
│   │       └── feature_distributions.png
│   │
│   └── metadata/
│       ├── dataset_info.csv     # Sample metadata
│       ├── dataset_stats.json   # Statistics
│       └── train_test_split.json # 80/20 split
│
└── backend/scripts/
    ├── phase2_data_organizer.py
    ├── phase2_feature_extractor.py
    ├── phase2_opensmile_extractor.py
    └── phase2_validate_data.py
```

---

## 🎓 Understanding the Features

### Spectral Features (50+)
- **MFCCs**: Voice characteristics, like a "fingerprint"
- **Spectral Centroid**: Brightness of voice (Hz)
- **Spectral Rolloff**: Frequency distribution
- **Spectral Bandwidth**: Spread of frequencies
- **Spectral Contrast**: Energy distribution across bands
- **Spectral Flatness**: Tonality vs noise
- **Zero Crossing Rate**: Voice quality indicator

### Temporal Features (25+)
- **RMS Energy**: Volume and variation
- **Tempo**: Speech rhythm
- **Onset Strength**: Speech dynamics
- **Duration**: Total speaking time

### Pitch Features (10+)
- **Pitch Mean/Std**: Average voice frequency
- **Pitch Range**: Voice flexibility
- **Pitch Variation**: Monotone vs varied speech

### Voice Quality (15+)
- **HNR**: Harmonics-to-Noise Ratio
- **Jitter**: Pitch stability
- **Shimmer**: Amplitude stability
- **Spectral Entropy**: Voice complexity

### Speech Timing (20+)
- **Pause Count**: Number of pauses
- **Pause Duration**: Length of pauses
- **Speech Ratio**: Speaking vs silence
- **Speech Rate**: Words per minute
- **Pause Density**: Pauses per minute

---

## 🔧 Troubleshooting

### Issue: "No audio files found"

**Solution:**
```bash
# Check if files exist
ls data/raw_audio/alzheimer/
ls data/raw_audio/healthy/

# If empty, add your .wav files there
```

---

### Issue: "librosa not found"

**Solution:**
```bash
pip install librosa soundfile
```

---

### Issue: "Features have missing values"

**Solution:**
This is normal for some features. The SVM trainer will handle this automatically by:
1. Filling missing values with mean
2. Removing zero-variance features
3. Scaling all features

---

### Issue: "Class imbalance detected"

**Solution:**
The SVM trainer uses `class_weight='balanced'` to handle imbalance automatically.

---

### Issue: "OpenSMILE not working"

**Solution:**
OpenSMILE is optional. You can skip it and use only librosa features:
```bash
# Just use the basic extractor
python backend/scripts/phase2_feature_extractor.py
```

---

## 📈 Performance Tips

### For Faster Processing

1. **Use fewer features:**
   - Edit `phase2_feature_extractor.py`
   - Comment out feature groups you don't need

2. **Process in parallel:**
   - Modify scripts to use `multiprocessing`
   - Process multiple files simultaneously

3. **Use smaller audio files:**
   - Trim to 30-60 seconds
   - Already done by the organizer

---

### For Better Accuracy

1. **Use all features:**
   - Run both librosa and OpenSMILE extractors
   - Use `features_combined.csv`

2. **Clean your data:**
   - Remove low-quality recordings
   - Ensure balanced classes
   - Remove outliers if needed

3. **Add more samples:**
   - More data = better model
   - Aim for 100+ samples per class

---

## ✅ Checklist

Before proceeding to Phase 3 (SVM Training):

- [ ] Audio files organized in `data/raw_audio/`
- [ ] Metadata created in `data/metadata/dataset_info.csv`
- [ ] Features extracted in `data/features/features.csv`
- [ ] Dataset validated (no critical issues)
- [ ] Visualizations created
- [ ] At least 50+ samples per class
- [ ] Features file has ~150 columns
- [ ] No missing labels

---

## 🎯 Next Steps

Once Phase 2 is complete:

```bash
# Verify everything is ready
python backend/scripts/phase2_validate_data.py

# If validation passes, proceed to Phase 3
python backend/scripts/svm_model_trainer.py --data-dir data/features
```

---

## 📞 Quick Commands Reference

```bash
# Step 1: Organize data
python backend/scripts/phase2_data_organizer.py

# Step 2: Extract features (required)
python backend/scripts/phase2_feature_extractor.py

# Step 2b: Extract OpenSMILE features (optional)
python backend/scripts/phase2_opensmile_extractor.py

# Step 3: Validate dataset
python backend/scripts/phase2_validate_data.py

# Check what you have
ls data/features/
cat data/features/feature_names.txt
head data/features/features.csv
```

---

## 🎉 Success Criteria

You're ready for Phase 3 when you see:

```
✅ DATASET IS READY FOR TRAINING!

Files created:
- data/features/features.csv (150 samples × 147 features)
- data/metadata/dataset_info.csv (150 rows)
- data/metadata/train_test_split.json

Next: Phase 3 - SVM Model Training
```

---

**🚀 Phase 2 Complete! Ready for SVM Training!**
