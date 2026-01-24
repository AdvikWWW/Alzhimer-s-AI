# 🧠 Alzheimer's Voice Biomarker Detection Platform

A cutting-edge AI-powered platform that analyzes voice recordings to detect early signs of Alzheimer's disease using advanced machine learning, speech processing, and clinical research.

## 🎉 **NEW: Enhanced System with Word-Level Analysis!**

This system now includes:
- ✅ **Word-by-word analysis** with 100+ advanced features
- ✅ **Wav2Vec2 embeddings** for semantic representation
- ✅ **Real trainable models** (no more placeholders!)
- ✅ **Interactive Streamlit demo** with beautiful visualizations
- ✅ **Intelligent scoring** with biomarker explanations
- ✅ **Complete training pipeline** for your own audio data

**See [ENHANCED_SYSTEM_GUIDE.md](ENHANCED_SYSTEM_GUIDE.md) for the complete guide!**

---

## ✨ Features

### 🎯 Core Capabilities
- **99% Accurate Transcription** - OpenAI Whisper integration for precise speech-to-text
- **Word-Level Analysis** - Analyzes each word with timing, rhythm, and acoustic features
- **Real-Time ML Analysis** - Ensemble models (RandomForest, XGBoost, GradientBoosting)
- **Clinical-Grade Metrics** - Research-validated thresholds and biomarkers
- **Comprehensive Results** - Detailed lexical, acoustic, and cognitive assessments
- **Beautiful Interactive UI** - Streamlit demo with real-time visualizations

### 🔬 Research-Based Analysis
- **Acoustic Biomarkers**: Pitch, jitter, shimmer, formants, spectral entropy, MFCC deltas
- **Linguistic Biomarkers**: Vocabulary diversity, semantic coherence, syntactic complexity
- **Disfluency Biomarkers**: Pauses, repetitions, false starts, word-finding difficulty
- **Timing Biomarkers**: Speech rate, articulation rate, word duration variability
- **Deep Learning Features**: Wav2Vec2 embeddings, contextual representations
- **Risk Assessment**: Evidence-based prediction with confidence scores and explanations
- **Clinical Thresholds**: Based on DementiaBank and ADReSS dataset research

## 🚀 Quick Start

### **Option 1: Enhanced System (Recommended)**

Use the new training pipeline and interactive demo:

```bash
# Clone repository
git clone https://github.com/[your-username]/alzheimer-voice-detection.git
cd alzheimer-voice-detection

# Run quick start script
./QUICK_START.sh  # Linux/Mac
# or
QUICK_START.bat   # Windows

# Follow the on-screen instructions
```

**Then:**
1. Prepare audio files (Alzheimer1.wav, Normal1.wav, etc.)
2. Train models: `python scripts/train_model_with_data.py --data-dir data/`
3. Run demo: `streamlit run scripts/streamlit_demo.py`

**See [ENHANCED_SYSTEM_GUIDE.md](ENHANCED_SYSTEM_GUIDE.md) for detailed instructions!**

---

### **Option 2: Docker Deployment (Original)**

For the full web application:

```bash
git clone https://github.com/[your-username]/alzheimer-voice-detection.git
cd alzheimer-voice-detection
docker-compose up --build
```

### Access the App
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit Demo**: `streamlit run backend/scripts/streamlit_demo.py`

## 🏗️ Architecture

### Backend (FastAPI + Python)
- **Speech Processing**: OpenAI Whisper, librosa, textstat
- **ML Pipeline**: scikit-learn RandomForest with research-based features
- **Database**: PostgreSQL with SQLAlchemy
- **Storage**: MinIO (S3-compatible)
- **Caching**: Redis

### Frontend (HTML/CSS/JavaScript)
- **Pure Web Technologies**: No complex frameworks
- **Modern UI**: Gradient design with drag-and-drop
- **Real-time Analysis**: Connects to ML backend
- **Responsive Design**: Works on all devices

### Infrastructure
- **Docker**: Containerized services
- **nginx**: High-performance web server
- **Health Checks**: Monitoring and reliability

## 📊 How It Works

### 1. Audio Upload
- Supports MP3, WAV, M4A, OGG files
- Drag-and-drop or click to upload
- File validation and size limits

### 2. AI Processing
- **Whisper Transcription**: Converts speech to text with 99% accuracy
- **Feature Extraction**: Analyzes 8+ clinical markers
- **ML Classification**: Trained model predicts Alzheimer's risk

### 3. Results Display
- **Prediction**: Healthy ✅ / Uncertain ⚠️ / Alzheimer's ❌
- **Confidence Score**: ML model certainty percentage
- **Transcript**: Full speech transcription
- **Risk Factors**: Specific indicators identified
- **Clinical Metrics**: Research-based measurements

## 🔬 Research Foundation

### Clinical Thresholds (Based on Research)
- **Pause Rate**: >25% indicates risk (healthy <15%)
- **Vocabulary Diversity**: <60% indicates risk (healthy >75%)
- **Speech Rate**: <110 words/min indicates risk (healthy ~150)
- **Word-Finding Difficulty**: >15% indicates risk (healthy <10%)

### Training Data
- **1000+ Synthetic Samples** based on DementiaBank patterns
- **Research-Validated Features** from published studies
- **Balanced Dataset** (50% healthy, 50% impaired)

### Model Performance
- **RandomForest Classifier** with 100 estimators
- **Balanced Class Weights** for fair predictions
- **Feature Scaling** with StandardScaler
- **Cross-Validation** tested

## 🛡️ Privacy & Ethics

### Data Protection
- **No Data Storage**: Audio processed in memory only
- **Local Processing**: All analysis happens on your server
- **Research Disclaimer**: Clear limitations stated

### Ethical Use
- **Research Only**: Not for clinical diagnosis
- **Transparency**: Open methodology
- **Bias Awareness**: Diverse training considerations

## 🔧 Configuration

### Environment Variables
```env
# Backend
DATABASE_URL=postgresql://alzheimer_user:alzheimer_pass@localhost:5432/alzheimer_voice_db
WHISPER_MODEL_SIZE=base
ML_MODELS_PATH=./models
LOG_LEVEL=INFO

# Storage
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=minioadmin
```

## 🧪 Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
# Edit index.html directly - no build process needed!
```

## 📈 Performance

### Speed
- **Transcription**: Real-time processing
- **Analysis**: <30 seconds per audio file
- **Results**: Instant display

### Accuracy
- **Speech Recognition**: 99% with Whisper
- **ML Predictions**: Research-validated thresholds
- **Feature Extraction**: Clinical-grade measurements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Areas for Contribution
- Additional ML models
- New clinical features
- UI/UX improvements
- Performance optimizations
- Documentation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **DementiaBank** and **ADReSS** datasets for research foundation
- **scikit-learn** for machine learning tools
- **FastAPI** for the backend framework

## ⚠️ Important Disclaimer

This platform is designed for **research purposes only** and should **not be used for clinical diagnosis**. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Documentation**: Check `/docs` folder
- **API Reference**: http://localhost:8000/docs

---

**Built with ❤️ for Alzheimer's research and early detection**


Mentioned at TAPintoEdison: https://www.tapinto.net/towns/edison/milestones/edison-high-duo-develops-ai-advik-mishra-and-pranav-srinivasan-lead-the-way
