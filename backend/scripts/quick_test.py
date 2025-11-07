#!/usr/bin/env python3
"""
Quick Test Script - Verify Installation and Basic Functionality
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required packages are installed"""
    print("🔍 Testing imports...")
    
    tests = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'librosa': 'Librosa',
        'sklearn': 'Scikit-learn',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'whisperx': 'WhisperX',
        'spacy': 'spaCy',
        'streamlit': 'Streamlit',
        'plotly': 'Plotly'
    }
    
    results = {}
    for module, name in tests.items():
        try:
            __import__(module)
            results[name] = '✅'
            print(f"  ✅ {name}")
        except ImportError as e:
            results[name] = '❌'
            print(f"  ❌ {name} - {str(e)}")
    
    return all(v == '✅' for v in results.values())


def test_services():
    """Test if services can be initialized"""
    print("\n🔍 Testing services...")
    
    try:
        from app.services.audio_processor import AudioProcessor
        audio_processor = AudioProcessor()
        print("  ✅ AudioProcessor initialized")
    except Exception as e:
        print(f"  ❌ AudioProcessor failed: {str(e)}")
        return False
    
    try:
        from app.services.asr_service import ASRService
        asr_service = ASRService()
        print("  ✅ ASRService initialized")
    except Exception as e:
        print(f"  ❌ ASRService failed: {str(e)}")
        return False
    
    try:
        from app.services.disfluency_analyzer import DisfluencyAnalyzer
        disfluency_analyzer = DisfluencyAnalyzer()
        print("  ✅ DisfluencyAnalyzer initialized")
    except Exception as e:
        print(f"  ❌ DisfluencyAnalyzer failed: {str(e)}")
        return False
    
    try:
        from app.services.lexical_semantic_analyzer import LexicalSemanticAnalyzer
        lexical_analyzer = LexicalSemanticAnalyzer()
        print("  ✅ LexicalSemanticAnalyzer initialized")
    except Exception as e:
        print(f"  ❌ LexicalSemanticAnalyzer failed: {str(e)}")
        return False
    
    return True


def test_spacy_model():
    """Test if spaCy English model is installed"""
    print("\n🔍 Testing spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("  ✅ spaCy English model loaded")
        return True
    except OSError:
        print("  ❌ spaCy English model not found")
        print("     Install with: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"  ❌ spaCy test failed: {str(e)}")
        return False


def test_gpu():
    """Test GPU availability"""
    print("\n🔍 Testing GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ⚠️  No GPU available (will use CPU)")
            return True
    except Exception as e:
        print(f"  ❌ GPU test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧠 ALZHEIMER'S VOICE DETECTION - SYSTEM TEST")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test services
    results.append(("Services", test_services()))
    
    # Test spaCy model
    results.append(("spaCy Model", test_spacy_model()))
    
    # Test GPU
    results.append(("GPU", test_gpu()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now:")
        print("  1. Run training: python scripts/train_model_with_data.py --data-dir <path>")
        print("  2. Run debug: python scripts/debug_model_pipeline.py --audio-files <files>")
        print("  3. Run demo: streamlit run scripts/streamlit_demo.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements_enhanced.txt")
        print("  - Download spaCy model: python -m spacy download en_core_web_sm")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
