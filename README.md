# Alzheimer's Voice Biomarker Research Platform

A comprehensive web application for detecting Alzheimer's disease biomarkers from voice recordings, designed for clinical research use.

## Features

- **Structured Recording Tasks**: Guided participant sessions including narrative, picture description, semantic fluency, reading passages, and more
- **Advanced Audio Processing**: Multi-stage preprocessing with VAD, diarization, and multi-engine ASR
- **Clinical Biomarkers**: Extraction of disfluencies, acoustic features, lexical-semantic markers, and speech timing
- **ML Inference**: Ensemble models trained on DementiaBank and ADReSS datasets
- **Research Reports**: Interactive timelines, structured analysis, and calibrated risk probabilities
- **Human QA**: Quality assurance workflow for low-confidence cases

## Architecture

### Frontend
- React 18 with TypeScript
- Tailwind CSS for styling
- Web Audio API for recording
- Real-time audio visualization

### Backend
- FastAPI with Python 3.9+
- PostgreSQL for metadata storage
- S3-compatible storage for audio files
- Docker deployment ready

### ML Pipeline
- WhisperX for ASR and forced alignment
- PyAnnote for diarization
- Librosa/Parselmouth for acoustic analysis
- Sentence-transformers for semantic analysis
- PyTorch/scikit-learn for inference

## Clinical Validation

Based on established research:
- López-de Ipiña et al. (2013): Speech disfluencies
- Saeedi et al. (2024): Acoustic features and voice quality
- Favaro et al. (2023): Lexical-semantic coherence
- Yang et al. (2022): Speech timing and fluency

## Setup

### Prerequisites
- Node.js 18+
- Python 3.9+
- Docker (optional)
- PostgreSQL
- S3-compatible storage

### Installation

1. Clone the repository
2. Install frontend dependencies: `cd frontend && npm install`
3. Install backend dependencies: `cd backend && pip install -r requirements.txt`
4. Set up environment variables (see `.env.example`)
5. Run database migrations
6. Start the development servers

### Development
```bash
# Frontend
cd frontend && npm run dev

# Backend
cd backend && uvicorn app.main:app --reload
```

## Research Use Only

This application is designed for research purposes only and should not be used for clinical diagnosis. All outputs include appropriate disclaimers and uncertainty measures.

## License

MIT License - See LICENSE file for details
