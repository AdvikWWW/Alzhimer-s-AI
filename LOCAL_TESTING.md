# Local Testing Guide - Validate Platform Accuracy

## 🎯 Testing Strategy

### What We're Testing
- Audio recording quality and functionality
- User interface responsiveness 
- Task completion workflow
- Chart and visualization accuracy
- Mobile compatibility
- Performance under load

## 🚀 Quick Frontend Testing (5 minutes)

### Option 1: Frontend-Only Testing
```bash
# Install Node.js from nodejs.org first
cd frontend
npm install
npm run dev
```
**Opens at:** http://localhost:5173

**What you can test:**
- ✅ Audio recording interface
- ✅ All 7 recording tasks
- ✅ Navigation and UI/UX
- ✅ Charts with mock data
- ✅ Admin dashboard
- ✅ Mobile responsiveness

### Option 2: Static Build Testing
```bash
cd frontend
npm run build
npm run preview
```
**Opens at:** http://localhost:4173

Tests the production build locally.

## 🔬 Comprehensive Testing Setup

### Option 3: Full Stack with Docker (Recommended)
```bash
# Install Docker Desktop first
docker-compose up --build
```

**Services Available:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Database: PostgreSQL on port 5432
- Storage: MinIO on http://localhost:9001

**What you can test:**
- ✅ Complete audio processing pipeline
- ✅ Real ML biomarker extraction
- ✅ Database storage and retrieval
- ✅ File upload and storage
- ✅ Full API functionality

### Option 4: Manual Backend Setup
```bash
# Terminal 1: Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m app.core.init_db
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

## 📋 Testing Checklist

### Core Functionality Tests
- [ ] **Audio Recording**
  - Record 30-second sample
  - Test pause/resume functionality
  - Verify audio levels display
  - Test on different browsers (Chrome, Safari, Firefox)

- [ ] **Recording Tasks**
  - Complete narrative task
  - Test picture description
  - Try semantic fluency task
  - Verify task navigation works

- [ ] **User Interface**
  - Check responsive design on mobile
  - Test all buttons and interactions
  - Verify charts load correctly
  - Check admin dashboard access

- [ ] **Performance**
  - Page load times (<3 seconds)
  - Audio recording starts quickly
  - Smooth navigation between tasks
  - Charts render without lag

### Accuracy Validation Tests

#### Test 1: Audio Quality Assessment
```bash
# Record samples and check:
- Clear audio capture (no distortion)
- Proper noise cancellation
- Consistent volume levels
- Format compatibility (WAV/MP3)
```

#### Test 2: UI/UX Validation
```bash
# Navigate through complete workflow:
1. Home page → Start session
2. Fill participant form
3. Complete 2-3 recording tasks
4. View results page
5. Check admin dashboard
```

#### Test 3: Cross-Browser Testing
```bash
# Test on multiple browsers:
- Chrome (primary)
- Safari (Mac/iOS)
- Firefox
- Edge
- Mobile browsers
```

#### Test 4: Mobile Responsiveness
```bash
# Test on different screen sizes:
- iPhone (375px width)
- iPad (768px width)
- Desktop (1200px+ width)
```

## 🔧 Testing Tools and Commands

### Performance Testing
```bash
# Build size analysis
cd frontend
npm run build
npx vite-bundle-analyzer dist

# Lighthouse audit
npx lighthouse http://localhost:5173 --view
```

### Network Testing
```bash
# Test with slow network
# Chrome DevTools → Network → Slow 3G

# Test offline functionality
# Chrome DevTools → Application → Service Workers
```

### Audio Testing Script
```javascript
// Test in browser console
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    console.log('✅ Audio access granted');
    stream.getTracks().forEach(track => track.stop());
  })
  .catch(err => console.error('❌ Audio access denied:', err));
```

## 📊 Expected Results

### Frontend-Only Testing
- **Load Time:** <2 seconds
- **Audio Recording:** Works in HTTPS/localhost
- **UI Responsiveness:** Smooth on mobile/desktop
- **Charts:** Display with mock data
- **Navigation:** All pages accessible

### Full Stack Testing
- **API Response Time:** <500ms for most endpoints
- **File Upload:** <5 seconds for 5MB audio file
- **ML Processing:** <30 seconds for biomarker extraction
- **Database Queries:** <100ms average

## 🐛 Common Issues and Solutions

### Audio Recording Issues
```bash
# Issue: "Permission denied" for microphone
# Solution: Ensure HTTPS or localhost, check browser permissions

# Issue: No audio levels showing
# Solution: Check Web Audio API support, try different browser

# Issue: Recording stops unexpectedly
# Solution: Check for memory limits, reduce recording quality
```

### Performance Issues
```bash
# Issue: Slow page loads
# Solution: Check network tab, optimize images, enable compression

# Issue: Charts not rendering
# Solution: Check console for errors, verify data format

# Issue: Mobile layout broken
# Solution: Test CSS media queries, check viewport settings
```

### Backend Connection Issues
```bash
# Issue: CORS errors
# Solution: Check backend CORS settings, verify URLs match

# Issue: 500 server errors
# Solution: Check backend logs, verify database connection

# Issue: File upload fails
# Solution: Check file size limits, storage configuration
```

## 📱 Mobile Testing Checklist

### iOS Safari
- [ ] Audio recording works
- [ ] Touch interactions smooth
- [ ] Charts display correctly
- [ ] No horizontal scrolling

### Android Chrome
- [ ] Microphone permissions work
- [ ] Responsive design adapts
- [ ] Performance acceptable
- [ ] All features accessible

## 🎯 Accuracy Benchmarks

### Expected Performance Metrics
- **Page Load:** <3 seconds on 3G
- **Audio Recording Start:** <1 second
- **Task Completion:** <2 minutes per task
- **Results Display:** <5 seconds
- **Mobile Usability:** 95%+ score

### Quality Indicators
- **Audio Quality:** Clear, no distortion
- **UI Responsiveness:** 60fps animations
- **Cross-browser Compatibility:** 95%+
- **Mobile Compatibility:** Works on iOS 12+, Android 8+
- **Accessibility:** WCAG 2.1 AA compliance

## 🚀 Quick Testing Commands

```bash
# 1. Quick frontend test
cd frontend && npm install && npm run dev

# 2. Production build test  
cd frontend && npm run build && npm run preview

# 3. Full stack test (with Docker)
docker-compose up --build

# 4. Performance audit
npx lighthouse http://localhost:5173 --view

# 5. Mobile simulation
# Chrome DevTools → Toggle Device Toolbar
```

## ✅ Ready to Publish Criteria

- [ ] All core features work in 3+ browsers
- [ ] Mobile experience is smooth
- [ ] Audio recording works reliably
- [ ] Page load times <3 seconds
- [ ] No console errors
- [ ] Charts and visualizations display correctly
- [ ] Admin dashboard functions properly
- [ ] Responsive design works on all screen sizes

Once you've validated these locally, you're ready to deploy with confidence! 🎉
