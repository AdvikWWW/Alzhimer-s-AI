# Alzheimer's Voice Biomarker Platform - Setup Guide

## Prerequisites

### 1. Install Node.js and npm
```bash
# Option 1: Download from official website
# Visit https://nodejs.org/ and download the LTS version

# Option 2: Using Homebrew (if available)
brew install node

# Option 3: Using nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install --lts
nvm use --lts
```

### 2. Install Python 3.8+ and pip
```bash
# Check if Python is installed
python3 --version

# If not installed, use Homebrew or download from python.org
brew install python
```

## Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python -m app.core.init_db
```

6. Start the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

## Development Workflow

1. **Frontend Development**: Run `npm run dev` in the frontend directory
2. **Backend Development**: Run `uvicorn app.main:app --reload` in the backend directory
3. **Database**: Ensure PostgreSQL is running locally or use Docker
4. **Testing**: Run `npm test` (frontend) and `pytest` (backend)

## Environment Configuration

### Frontend (.env)
```
VITE_API_BASE_URL=http://localhost:8000
VITE_ENVIRONMENT=development
```

### Backend (.env)
```
DATABASE_URL=postgresql://user:password@localhost:5432/alzheimer_voice
SECRET_KEY=your-secret-key-here
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=minioadmin
S3_BUCKET_NAME=alzheimer-voice-data
```

## Docker Setup (Alternative)

If you prefer using Docker:

```bash
# Build and start all services
docker-compose up --build

# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# Database: PostgreSQL on port 5432
# Storage: MinIO on port 9000
```

## Troubleshooting

### Common Issues

1. **Node.js not found**: Install Node.js from nodejs.org or use a package manager
2. **Python dependencies fail**: Ensure you have Python 3.8+ and pip installed
3. **Database connection errors**: Check PostgreSQL is running and credentials are correct
4. **CORS issues**: Verify frontend and backend URLs in environment variables

### Development Tips

- Use `npm run build` to create production build
- Use `pytest --cov` for backend test coverage
- Check `npm run lint` for frontend code quality
- Monitor logs in both frontend and backend terminals

## Next Steps

1. Install Node.js and npm
2. Run `npm install` in the frontend directory
3. Install Python dependencies in the backend
4. Set up PostgreSQL database
5. Configure environment variables
6. Start both frontend and backend servers

The platform includes:
- ✅ Audio recording interface with real-time visualization
- ✅ Structured cognitive assessment tasks
- ✅ Advanced audio processing pipeline (VAD, ASR, diarization)
- ✅ Clinical biomarker extraction (acoustic, lexical, disfluency)
- ✅ Interactive results dashboard with charts and timelines
- ✅ Admin panel for quality control and system monitoring
- 🔄 ML inference engine (in development)
- 🔄 Secure storage and database integration
- 🔄 Production deployment configuration
