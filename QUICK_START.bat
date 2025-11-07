@echo off
REM Quick Start Script for Enhanced Alzheimer's Voice Detection (Windows)

echo ============================================================
echo 🧠 ALZHEIMER'S VOICE DETECTION - QUICK START
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Navigate to backend directory
cd backend

REM Step 1: Install dependencies
echo.
echo ============================================================
echo 📦 Step 1: Installing Dependencies
echo ============================================================
echo.

if not exist "venv" (
    echo ⚠️  Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
)

echo ⚠️  Activating virtual environment...
call venv\Scripts\activate.bat

echo ⚠️  Installing base requirements...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo ⚠️  Installing enhanced requirements...
pip install -r requirements_enhanced.txt --quiet

echo ✅ All dependencies installed

REM Step 2: Download spaCy model
echo.
echo ============================================================
echo 📚 Step 2: Downloading spaCy Model
echo ============================================================
echo.

python -m spacy download en_core_web_sm
echo ✅ spaCy model downloaded

REM Step 3: Run system test
echo.
echo ============================================================
echo 🔍 Step 3: Running System Test
echo ============================================================
echo.

python scripts\quick_test.py

if errorlevel 1 (
    echo ❌ System test failed. Please check the errors above.
    pause
    exit /b 1
)

echo ✅ System test passed!

REM Step 4: Instructions
echo.
echo ============================================================
echo 🎉 SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo.
echo 1. Prepare your audio data:
echo    - Create a directory with audio files
echo    - Name files: Alzheimer1.wav, Alzheimer2.wav, ..., Normal1.wav, Normal2.wav, ...
echo.
echo 2. Debug feature extraction:
echo    python scripts\debug_model_pipeline.py --audio-files path\to\Alzheimer1.wav path\to\Normal1.wav
echo.
echo 3. Train models:
echo    python scripts\train_model_with_data.py --data-dir path\to\audio\files --output-dir models\
echo.
echo 4. Run interactive demo:
echo    streamlit run scripts\streamlit_demo.py
echo.
echo For detailed instructions, see:
echo   - ENHANCED_SYSTEM_GUIDE.md
echo   - IMPLEMENTATION_SUMMARY.md
echo.
echo ============================================================
echo.
echo ✅ Ready to use! 🚀
echo.

pause
