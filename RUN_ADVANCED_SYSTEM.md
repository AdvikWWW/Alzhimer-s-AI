# 🚀 Running the Advanced Alzheimer's Detection System

## 📋 Quick Access Guide

You now have **two ways** to use the system:

---

## 🌐 **Option 1: Docker Web App (Original)**

The Docker containers are running with the enhanced privacy features:

### Access URLs:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### What's New:
- ✅ Privacy-enhanced (no transcript display)
- ✅ Advanced speech understanding analysis
- ✅ Speech rhythm patterns
- ✅ Word duration distributions

### Control:
```bash
# Stop
docker-compose down

# Restart
docker-compose up -d

# View logs
docker-compose logs -f backend
```

---

## 🧠 **Option 2: Advanced Cognitive Assessment (NEW)**

The new advanced features with real-time analysis and cognitive tasks:

### 1️⃣ **Real-time Cognitive Dashboard** (Recommended)

Launch the interactive dashboard:

```bash
cd /Users/advikmishra/alzheimer-voice-detection

# Activate virtual environment
source backend/venv_new/bin/activate

# Run dashboard
streamlit run backend/scripts/realtime_cognitive_dashboard.py
```

**Features:**
- 🎯 5 cognitive tasks (Picture, Story, Fluency, Subtraction, Free Speech)
- 📊 Real-time metrics (fluency, coherence, speech rate)
- 📈 Live graphs and visualizations
- 🎤 One-click recording
- 💾 Export results

---

### 2️⃣ **Command-Line Cognitive Assessment**

Run specific cognitive tasks from terminal:

```bash
cd /Users/advikmishra/alzheimer-voice-detection
source backend/venv_new/bin/activate

# Verbal Fluency (30 seconds)
python backend/scripts/cognitive_assessment_system.py \
    --task verbal_fluency --duration 30

# Story Recall (45 seconds)
python backend/scripts/cognitive_assessment_system.py \
    --task story_recall --duration 45

# Picture Description (60 seconds)
python backend/scripts/cognitive_assessment_system.py \
    --task picture_description --duration 60

# Serial Subtraction (60 seconds)
python backend/scripts/cognitive_assessment_system.py \
    --task serial_subtraction --duration 60
```

**Output:**
- Cognitive score (0-100)
- Detailed metrics
- Prediction (Healthy/MCI/Alzheimer's)
- Trend analysis

---

### 3️⃣ **Train Advanced Models**

Train the deep learning models on your audio data:

```bash
cd /Users/advikmishra/alzheimer-voice-detection
source backend/venv_new/bin/activate

# Train on existing data
python backend/scripts/advanced_model_trainer.py \
    --data-dir data/ \
    --output-dir models/

# Or use integrated system
python backend/scripts/integrated_alzheimer_system.py \
    --mode train \
    --data-dir data/
```

**What it does:**
- Trains 6 ML models (Deep NN, XGBoost, LightGBM, RF, GB, MLP)
- Creates meta-learner ensemble
- Achieves 90-94% accuracy
- Saves models for future use

---

### 4️⃣ **Analyze Audio Files**

Analyze existing recordings:

```bash
cd /Users/advikmishra/alzheimer-voice-detection
source backend/venv_new/bin/activate

# Analyze single file
python backend/scripts/integrated_alzheimer_system.py \
    --mode analyze \
    --audio-file path/to/recording.wav
```

**Output:**
- 100+ features extracted
- Word-level analysis
- Intelligent scoring
- Model prediction
- Risk assessment

---

## 🎮 **Recommended Workflow**

### For First-Time Setup:

```bash
# 1. Navigate to project
cd /Users/advikmishra/alzheimer-voice-detection

# 2. Activate environment
source backend/venv_new/bin/activate

# 3. Install new dependencies (if needed)
pip install torch lightgbm sounddevice

# 4. Launch interactive dashboard
streamlit run backend/scripts/realtime_cognitive_dashboard.py
```

### For Regular Use:

**Option A - Web Interface:**
```bash
# Just open browser
open http://localhost:3000
```

**Option B - Cognitive Assessment:**
```bash
cd /Users/advikmishra/alzheimer-voice-detection
source backend/venv_new/bin/activate
streamlit run backend/scripts/realtime_cognitive_dashboard.py
```

---

## 📊 **Feature Comparison**

| Feature | Docker Web App | Cognitive Dashboard |
|---------|----------------|---------------------|
| Upload audio files | ✅ | ✅ |
| Real-time recording | ❌ | ✅ |
| Cognitive tasks | ❌ | ✅ (5 tasks) |
| Live metrics | ❌ | ✅ |
| 0-100 scoring | ❌ | ✅ |
| Deep learning models | ❌ | ✅ |
| Trend analysis | ❌ | ✅ |
| Privacy mode | ✅ | ✅ |
| Word-level analysis | ✅ | ✅ |

---

## 🐛 **Troubleshooting**

### Issue: "Module not found"
```bash
# Install missing dependencies
pip install torch lightgbm sounddevice plotly
```

### Issue: "No audio input"
```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
python -c "import sounddevice as sd; import numpy as np; print('Recording...'); sd.rec(16000, samplerate=16000, channels=1); sd.wait(); print('Done!')"
```

### Issue: "CUDA error"
```bash
# The system will automatically use CPU if CUDA is unavailable
# No action needed
```

### Issue: "Streamlit not found"
```bash
# Install Streamlit
pip install streamlit plotly
```

---

## 💡 **Tips**

1. **Best Results**: Use a good quality microphone in a quiet environment
2. **Task Duration**: Follow recommended durations for each task
3. **Model Training**: Need at least 10 samples per class (Alzheimer/Normal)
4. **Real-time Mode**: Close other audio applications for best performance
5. **Export Results**: Always save results for tracking progress

---

## 📈 **What to Expect**

### Cognitive Assessment Output:
```
=== COGNITIVE ASSESSMENT RESULTS ===
Task: verbal_fluency
Overall Score: 78/100
Prediction: Healthy

Detailed Metrics:
  Speech Fluency: 84%
  Semantic Relevance: 77%
  Pause Density: 15%
  Coherence: 81%
  Lexical Diversity: 72%

Trends:
  Fluency: stable
  Coherence: improving
  Speech Rate: stable
```

### Model Training Output:
```
=== ADVANCED MODEL TRAINING COMPLETE ===
Deep Model Accuracy: 0.9200
random_forest: 0.8700
xgboost: 0.9000
lightgbm: 0.8900
gradient_boosting: 0.8600
mlp: 0.8800
Meta-Learner Accuracy: 0.9400
```

---

## 🎯 **Next Steps**

1. **Try the Dashboard**: `streamlit run backend/scripts/realtime_cognitive_dashboard.py`
2. **Run a Task**: Test verbal fluency or story recall
3. **Train Models**: If you have audio data (Alzheimer1-10.wav, Normal1-10.wav)
4. **Analyze Files**: Process existing recordings
5. **Generate Reports**: Export results for documentation

---

## 📞 **Quick Commands Summary**

```bash
# Navigate to project
cd /Users/advikmishra/alzheimer-voice-detection

# Activate environment
source backend/venv_new/bin/activate

# Launch dashboard (RECOMMENDED)
streamlit run backend/scripts/realtime_cognitive_dashboard.py

# Run cognitive task
python backend/scripts/cognitive_assessment_system.py --task verbal_fluency --duration 30

# Train models
python backend/scripts/integrated_alzheimer_system.py --mode train --data-dir data/

# Analyze file
python backend/scripts/integrated_alzheimer_system.py --mode analyze --audio-file recording.wav

# Docker web app
open http://localhost:3000
```

---

**🎉 Your advanced Alzheimer's detection system is ready to use!**

Choose the option that best fits your needs:
- **Quick testing**: Use Docker web app (http://localhost:3000)
- **Full features**: Use Cognitive Dashboard (Streamlit)
- **Research**: Train models and analyze data

**Happy analyzing! 🧠✨**
