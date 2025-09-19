# Quick Start Guide - Get Your Website Live in 30 Minutes

## 🎯 Goal: Deploy Alzheimer's Voice Platform as a Live Website

### Step 1: Install Prerequisites (5 minutes)

**Install Node.js:**
1. Go to [nodejs.org](https://nodejs.org/)
2. Download the LTS version (18.x or higher)
3. Run the installer
4. Verify installation: `node --version` and `npm --version`

**Install Python (if not already installed):**
1. Go to [python.org](https://python.org/downloads/)
2. Download Python 3.8+ 
3. Run installer (check "Add to PATH")
4. Verify: `python3 --version`

### Step 2: Test Frontend Locally (10 minutes)

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Start development server
npm run dev
```

**Expected Result:** Frontend runs at http://localhost:5173

### Step 3: Deploy Frontend to Web (10 minutes)

**Option A: Netlify (Recommended)**
```bash
# Build the project
npm run build

# Install Netlify CLI
npm install -g netlify-cli

# Deploy
netlify deploy --prod --dir=dist
```

**Option B: Vercel**
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

**Option C: GitHub Pages**
1. Push code to GitHub repository
2. Go to repository Settings > Pages
3. Select source branch
4. Your site will be live at `https://yourusername.github.io/repo-name`

### Step 4: Deploy Backend (Optional for MVP)

**Quick Backend Deployment:**
```bash
# For Railway (easiest)
npm install -g @railway/cli
railway login
railway link
railway up
```

## 🚀 Fastest Path to Live Website

### Frontend-Only Demo (No Backend Required)
The frontend can run independently with mock data for demonstration purposes. This is perfect for:
- Showcasing the UI/UX
- Testing audio recording
- Demonstrating the workflow
- Getting feedback

**Deploy Steps:**
1. `cd frontend`
2. `npm install`
3. `npm run build`
4. `netlify deploy --prod --dir=dist`

**Result:** Live website in ~15 minutes!

### Full Stack Deployment
For complete functionality including ML analysis:
1. Deploy frontend (above)
2. Deploy backend to Railway/Render
3. Set up production database
4. Configure storage service

## 📱 Testing Without Installation

### Online Development Environments
1. **CodeSandbox**: Import GitHub repo, runs in browser
2. **Gitpod**: Full development environment in browser
3. **Replit**: Quick prototyping and testing

### Browser-Based Testing
The frontend is designed to work with mock data, so you can:
- Test audio recording
- Navigate through tasks
- View results dashboard
- Use admin panel

## 🎯 Recommended Deployment Strategy

### Phase 1: Frontend Demo (Today)
- Deploy frontend to Netlify/Vercel
- Use mock data for demonstration
- Share link for feedback
- Test on mobile devices

### Phase 2: Backend Integration (This Week)
- Deploy backend to Railway
- Connect to managed database
- Set up file storage
- Enable full ML pipeline

### Phase 3: Production (Next Week)
- Custom domain
- SSL certificates
- Performance optimization
- Monitoring and analytics

## 🔧 Environment Setup Commands

```bash
# Check if tools are installed
node --version
npm --version
python3 --version

# If not installed, download from:
# Node.js: https://nodejs.org/
# Python: https://python.org/downloads/

# Quick frontend test
cd frontend
npm install
npm run dev

# Quick build and deploy
npm run build
npx netlify-cli deploy --prod --dir=dist
```

## 🌐 Expected URLs After Deployment

- **Netlify**: `https://amazing-app-name.netlify.app`
- **Vercel**: `https://project-name.vercel.app`
- **Railway**: `https://project-name.railway.app`

## 📋 Pre-Launch Checklist

- [ ] Audio recording works in browser
- [ ] All pages load correctly
- [ ] Mobile responsive design works
- [ ] Charts and visualizations display
- [ ] No console errors
- [ ] HTTPS enabled
- [ ] Fast loading times (<3 seconds)

## 💡 Pro Tips

1. **Start with frontend-only** - Get something live quickly
2. **Use mock data initially** - Perfect for demos and feedback
3. **Mobile-first testing** - Most users will access on phones
4. **Share early and often** - Get feedback before full backend
5. **Monitor performance** - Use built-in analytics

## 🆘 If You Get Stuck

### Common Solutions
- **Node.js not found**: Restart terminal after installation
- **Permission errors**: Use `sudo` on Mac/Linux, run as admin on Windows
- **Build failures**: Check Node.js version (need 16+)
- **Deploy errors**: Check build output for specific errors

### Quick Help Commands
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Check for errors
npm run build
```

Your Alzheimer's Voice Biomarker Platform is ready to go live! 🚀
