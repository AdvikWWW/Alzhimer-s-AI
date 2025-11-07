#!/bin/bash

# Quick Start Script for Enhanced Alzheimer's Voice Detection
# This script helps you get started quickly

set -e  # Exit on error

echo "============================================================"
echo "🧠 ALZHEIMER'S VOICE DETECTION - QUICK START"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

print_success "Python 3 found: $(python3 --version)"

# Navigate to backend directory
cd backend

# Step 1: Install dependencies
echo ""
echo "============================================================"
echo "📦 Step 1: Installing Dependencies"
echo "============================================================"
echo ""

if [ ! -d "venv" ]; then
    print_warning "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

print_warning "Activating virtual environment..."
source venv/bin/activate

print_warning "Installing base requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

print_warning "Installing enhanced requirements..."
pip install -q -r requirements_enhanced.txt

print_success "All dependencies installed"

# Step 2: Download spaCy model
echo ""
echo "============================================================"
echo "📚 Step 2: Downloading spaCy Model"
echo "============================================================"
echo ""

python -m spacy download en_core_web_sm
print_success "spaCy model downloaded"

# Step 3: Run system test
echo ""
echo "============================================================"
echo "🔍 Step 3: Running System Test"
echo "============================================================"
echo ""

python scripts/quick_test.py

if [ $? -eq 0 ]; then
    print_success "System test passed!"
else
    print_error "System test failed. Please check the errors above."
    exit 1
fi

# Step 4: Instructions
echo ""
echo "============================================================"
echo "🎉 SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Prepare your audio data:"
echo "   - Create a directory with audio files"
echo "   - Name files: Alzheimer1.wav, Alzheimer2.wav, ..., Normal1.wav, Normal2.wav, ..."
echo ""
echo "2. Debug feature extraction:"
echo "   python scripts/debug_model_pipeline.py --audio-files path/to/Alzheimer1.wav path/to/Normal1.wav"
echo ""
echo "3. Train models:"
echo "   python scripts/train_model_with_data.py --data-dir path/to/audio/files --output-dir models/"
echo ""
echo "4. Run interactive demo:"
echo "   streamlit run scripts/streamlit_demo.py"
echo ""
echo "For detailed instructions, see:"
echo "  - ENHANCED_SYSTEM_GUIDE.md"
echo "  - IMPLEMENTATION_SUMMARY.md"
echo ""
echo "============================================================"
echo ""

print_success "Ready to use! 🚀"
