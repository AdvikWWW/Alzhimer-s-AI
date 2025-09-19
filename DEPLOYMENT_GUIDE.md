# Deployment Guide - Alzheimer's Voice Biomarker Platform

## 🚀 Local Testing Setup

### Option 1: Docker (Recommended for Full Testing)

1. **Install Docker Desktop**
   - Download from [docker.com](https://www.docker.com/products/docker-desktop/)
   - Ensure Docker Compose is included

2. **Run the Complete Stack**
   ```bash
   cd /Users/pranavsrinivasan/CascadeProjects/windsurf-project
   docker-compose up --build
   ```

3. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - MinIO Storage: http://localhost:9001 (admin/admin123)
   - Database: PostgreSQL on localhost:5432

### Option 2: Manual Setup (Development)

1. **Install Prerequisites**
   ```bash
   # Install Node.js 18+ from nodejs.org
   # Install Python 3.8+ from python.org
   # Install PostgreSQL from postgresql.org
   ```

2. **Setup Backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Copy and configure environment
   cp .env.example .env
   # Edit .env with your database credentials
   
   # Initialize database
   python -m app.core.init_db
   
   # Start backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   
   # Copy and configure environment
   cp .env.example .env
   # Edit .env to point to your backend URL
   
   # Start frontend
   npm run dev
   ```

## 🌐 Website Deployment

### Frontend Deployment Options

#### Option 1: Netlify (Recommended)
```bash
# Build the frontend
cd frontend
npm run build

# Deploy to Netlify
npx netlify-cli deploy --prod --dir=dist
```

**Netlify Configuration (`netlify.toml`):**
```toml
[build]
  publish = "dist"
  command = "npm run build"

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

#### Option 2: Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel --prod
```

#### Option 3: GitHub Pages
```bash
# Add to package.json
"homepage": "https://yourusername.github.io/alzheimer-voice-platform"

# Build and deploy
npm run build
npm run deploy
```

### Backend Deployment Options

#### Option 1: Railway (Recommended for MVP)
1. Connect GitHub repository to Railway
2. Set environment variables:
   ```
   DATABASE_URL=postgresql://...
   S3_ENDPOINT_URL=https://...
   SECRET_KEY=your-secret-key
   ```
3. Deploy automatically on push

#### Option 2: Render
1. Connect repository to Render
2. Configure build command: `pip install -r requirements.txt`
3. Configure start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### Option 3: AWS/Google Cloud
- Use container deployment with Docker
- Set up managed database (RDS/Cloud SQL)
- Configure S3/Cloud Storage

### Database & Storage Setup

#### Production Database Options
1. **Railway PostgreSQL** (easiest)
2. **Supabase** (PostgreSQL with extras)
3. **AWS RDS** (enterprise)
4. **Google Cloud SQL** (enterprise)

#### Storage Options
1. **AWS S3** (production-ready)
2. **Google Cloud Storage**
3. **Cloudflare R2** (S3-compatible, cheaper)

## 📋 Pre-Deployment Checklist

### Frontend Configuration
- [ ] Update `VITE_API_BASE_URL` to production backend URL
- [ ] Configure proper CORS origins in backend
- [ ] Test audio recording in production environment
- [ ] Verify all charts and components work
- [ ] Test admin dashboard functionality

### Backend Configuration
- [ ] Set secure `SECRET_KEY`
- [ ] Configure production database URL
- [ ] Set up production storage (S3/equivalent)
- [ ] Configure proper CORS origins
- [ ] Set up logging and monitoring
- [ ] Test all API endpoints

### Security & Performance
- [ ] Enable HTTPS for both frontend and backend
- [ ] Configure rate limiting
- [ ] Set up proper error handling
- [ ] Configure file upload limits
- [ ] Test with real audio files
- [ ] Verify ML model loading

## 🚀 Quick Deploy Commands

### Deploy Frontend to Netlify
```bash
cd frontend
npm run build
npx netlify-cli deploy --prod --dir=dist
```

### Deploy Backend to Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up
```

## 🔧 Environment Variables

### Frontend (.env)
```
VITE_API_BASE_URL=https://your-backend-url.com
VITE_ENVIRONMENT=production
VITE_APP_NAME=Alzheimer Voice Biomarker Platform
```

### Backend (.env)
```
DATABASE_URL=postgresql://user:pass@host:port/db
SECRET_KEY=your-super-secret-key-here
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY_ID=your-access-key
S3_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=alzheimer-voice-data
BACKEND_CORS_ORIGINS=["https://your-frontend-url.com"]
```

## 📱 Mobile App Considerations

### Future Mobile Deployment
1. **React Native**: Convert existing React components
2. **Capacitor**: Wrap web app as native app
3. **PWA**: Progressive Web App for mobile browsers

### Mobile-Specific Features
- Native audio recording APIs
- Offline capability
- Push notifications for reminders
- Biometric authentication

## 🔍 Testing Your Deployment

### Local Testing Checklist
- [ ] Audio recording works in browser
- [ ] All 7 recording tasks function properly
- [ ] File upload and processing works
- [ ] Charts and visualizations display correctly
- [ ] Admin dashboard is accessible
- [ ] Database connections work
- [ ] Storage service functions

### Production Testing
- [ ] HTTPS works correctly
- [ ] Audio recording works on mobile browsers
- [ ] File uploads work with production storage
- [ ] API responses are fast (<2s)
- [ ] All pages load quickly
- [ ] Error handling works properly

## 💡 Deployment Tips

1. **Start Simple**: Deploy frontend to Netlify first, backend to Railway
2. **Test Thoroughly**: Use the local Docker setup to test everything
3. **Monitor Performance**: Set up logging and monitoring from day one
4. **Security First**: Always use HTTPS and secure environment variables
5. **Scale Gradually**: Start with managed services, optimize later

## 🆘 Troubleshooting

### Common Issues
- **CORS Errors**: Check backend CORS configuration
- **Audio Not Working**: Ensure HTTPS is enabled
- **File Upload Fails**: Check storage service configuration
- **Database Connection**: Verify connection string and credentials
- **Build Failures**: Check Node.js/Python versions

### Debug Commands
```bash
# Check backend logs
docker-compose logs backend

# Check frontend build
npm run build

# Test API endpoints
curl https://your-backend-url.com/health
```
