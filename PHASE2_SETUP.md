# 📊 Phase 2: Data & Feature Extraction - Complete Guide

## Overview

This phase organizes 150+ voice recordings and extracts comprehensive audio features for SVM training.

---

## 📁 Folder Structure

```
alzheimer-voice-detection/
├── data/
│   ├── raw_audio/                    # Original recordings
│   │   ├── alzheimer/                # Alzheimer's patients (75+ files)
│   │   │   ├── alz_001.wav
│   │   │   ├── alz_002.wav
│   │   │   └── ...
│   │   └── healthy/                  # Healthy controls (75+ files)
│   │       ├── healthy_001.wav
│   │       ├── healthy_002.wav
│   │       └── ...
│   │
│   ├── processed/                    # Processed audio (normalized, trimmed)
│   │   ├── alzheimer/
│   │   └── healthy/
│   │
│   ├── features/                     # Extracted features
│   │   ├── features.csv              # Main feature dataset
│   │   ├── features.npy              # NumPy format
│   │   ├── librosa_features.csv      # Librosa-specific
│   │   └── opensmile_features.csv    # OpenSMILE-specific
│   │
│   └── metadata/
│       ├── dataset_info.csv          # Sample metadata
│       ├── train_test_split.json     # Train/test indices
│       └── feature_descriptions.json # Feature documentation
│
├── backend/
│   └── scripts/
│       ├── phase2_data_organizer.py       # Organize audio files
│       ├── phase2_feature_extractor.py    # Extract features
│       ├── phase2_opensmile_extractor.py  # OpenSMILE features
│       └── phase2_validate_data.py        # Validate dataset
│
└── models/
    └── svm/
        └── features/                 # Features used for training
```

---

## 📝 Metadata Format

### dataset_info.csv

| file_id | filename | label | age | gender | duration_sec | recording_date | task_type | quality_score | notes |
|---------|----------|-------|-----|--------|--------------|----------------|-----------|---------------|-------|
| alz_001 | alz_001.wav | Alzheimer | 72 | M | 45.3 | 2024-01-15 | verbal_fluency | 8.5 | Clear audio |
| alz_002 | alz_002.wav | Alzheimer | 68 | F | 38.7 | 2024-01-16 | picture_desc | 7.2 | Some noise |
| healthy_001 | healthy_001.wav | Healthy | 70 | M | 42.1 | 2024-01-15 | verbal_fluency | 9.0 | Excellent |

**Columns:**
- `file_id`: Unique identifier
- `filename`: Audio file name
- `label`: "Alzheimer" or "Healthy"
- `age`: Patient age
- `gender`: M/F
- `duration_sec`: Recording length
- `recording_date`: When recorded
- `task_type`: Type of cognitive task
- `quality_score`: Audio quality (1-10)
- `notes`: Additional information

---

## 🎵 Audio Requirements

### Format Specifications

```
Format: WAV (PCM)
Sample Rate: 16,000 Hz (16 kHz)
Channels: Mono (1 channel)
Bit Depth: 16-bit
Duration: 30-120 seconds recommended
File Size: ~1-4 MB per file
```

### Quality Criteria

✅ **Good Quality:**
- Clear speech
- Minimal background noise
- SNR > 20 dB
- No clipping/distortion

❌ **Poor Quality:**
- Heavy background noise
- Multiple speakers
- Severe distortion
- Very short (< 10 seconds)

---

## 🔬 Features to Extract

### 1. Librosa Features (40+ features)

#### Spectral Features
- **MFCCs** (13 coefficients + deltas + delta-deltas = 39)
- **Spectral Centroid** (mean, std, min, max)
- **Spectral Rolloff** (mean, std)
- **Spectral Bandwidth** (mean, std)
- **Spectral Contrast** (7 bands)
- **Spectral Flatness** (mean, std)
- **Zero Crossing Rate** (mean, std)

#### Temporal Features
- **RMS Energy** (mean, std, min, max)
- **Tempo** (estimated BPM)
- **Onset Strength** (mean, std)

#### Pitch Features
- **Pitch (F0)** (mean, std, min, max, range)
- **Pitch Variation** (coefficient of variation)

#### Rhythm Features
- **Rhythm Patterns** (autocorrelation)
- **Beat Tracking** (if applicable)

---

### 2. OpenSMILE Features (80+ features)

#### Prosodic Features
- **Pitch (F0)** statistics
- **Jitter** (pitch variation)
- **Shimmer** (amplitude variation)
- **Harmonics-to-Noise Ratio (HNR)**

#### Voice Quality
- **Formants** (F1, F2, F3, F4)
- **Formant Bandwidths**
- **Spectral Tilt**
- **Cepstral Peak Prominence (CPP)**

#### Energy Features
- **Loudness** (mean, std, range)
- **Energy in frequency bands**
- **Voice Activity Detection (VAD)**

#### Temporal Features
- **Speaking Rate** (syllables/second)
- **Pause Duration** (mean, total)
- **Pause Frequency**
- **Speech-to-Pause Ratio**

---

### 3. Custom Alzheimer's-Specific Features (30+ features)

#### Disfluency Markers
- **Hesitation Count** (um, uh, er)
- **Repetition Rate**
- **False Starts**
- **Self-Corrections**

#### Lexical Features
- **Word Count**
- **Unique Words**
- **Type-Token Ratio** (vocabulary richness)
- **Average Word Length**

#### Cognitive Biomarkers
- **Pause Density** (pauses per minute)
- **Speech Rate** (words per minute)
- **Articulation Rate** (excluding pauses)
- **Phonation Time** (total speaking time)

---

## 📊 Final Feature Set

### Total Features: ~150

| Category | Count | Examples |
|----------|-------|----------|
| **Spectral** | 50 | MFCCs, spectral centroid, rolloff |
| **Prosodic** | 30 | Pitch, jitter, shimmer, formants |
| **Temporal** | 25 | Energy, tempo, rhythm |
| **Linguistic** | 20 | Word count, TTR, hesitations |
| **Voice Quality** | 15 | HNR, CPP, spectral tilt |
| **Custom** | 10 | Pause density, speech rate |

**Total: ~150 features per recording**

---

## 🚀 Next Steps

1. ✅ Create folder structure
2. ✅ Set up metadata template
3. ✅ Install dependencies (librosa, opensmile)
4. ✅ Create feature extraction scripts
5. ✅ Process all 150+ recordings
6. ✅ Generate features.csv
7. ✅ Validate dataset
8. ✅ Ready for Phase 3 (SVM Training)

---

## 📦 Dependencies

```bash
# Audio processing
pip install librosa soundfile

# OpenSMILE
pip install opensmile

# Data processing
pip install numpy pandas scikit-learn

# Visualization
pip install matplotlib seaborn

# Progress tracking
pip install tqdm
```

---

## ⏱️ Estimated Time

- **Setup folders**: 5 minutes
- **Organize 150 files**: 15 minutes
- **Create metadata**: 20 minutes
- **Extract features**: 30-60 minutes (automated)
- **Validate data**: 10 minutes

**Total: ~1.5-2 hours** (mostly automated)

---

**Ready to create the scripts!**
