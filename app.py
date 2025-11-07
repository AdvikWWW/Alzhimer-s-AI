import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import sys
import os
import tempfile
import io
from run_model import AlzheimerDetector

# Set page config
st.set_page_config(
    page_title="Alzheimer's Voice Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alzheimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .healthy {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .audio-player {
        width: 100%;
        margin: 1rem 0;
    }
    .feature-title {
        color: #1e88e5;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🧠 Alzheimer's Voice Detection")
    st.markdown("---")
    
    # Initialize detector
    if 'detector' not in st.session_state:
        with st.spinner('Loading model... This may take a moment...'):
            try:
                st.session_state.detector = AlzheimerDetector()
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.stop()
    
    # Sidebar with info
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain-3.png", width=80)
        st.title("About")
        st.markdown("""
        This tool analyzes voice patterns to detect potential signs of Alzheimer's disease.
        
        **How it works:**
        1. Upload an audio file or record your voice
        2. The model analyzes 101 acoustic features
        3. Get instant results with confidence scores
        
        **Note:** This is a research tool and should not be used for clinical diagnosis.
        """)
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info("""
        - **Model:** SVM-RBF
        - **Accuracy:** 90%
        - **Trained on:** 50 real recordings
        - **Last updated:** Nov 2024
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload Audio")
        
        # Audio input options
        input_method = st.radio("Choose input method:", 
                              ["Upload audio file", "Record voice"],
                              horizontal=True)
        
        audio_file = None
        
        if input_method == "Upload audio file":
            audio_file = st.file_uploader("Upload an audio file (WAV, MP3, M4A)", 
                                        type=['wav', 'mp3', 'm4a'])
        else:
            audio_file = st.audio("Record your voice", 
                                format="audio/wav", 
                                start_prompt="Start recording",
                                stop_prompt="Stop recording")
        
        if audio_file is not None:
            # Save uploaded file to temp file
            try:
                # Determine file extension
                if hasattr(audio_file, 'name'):
                    file_ext = Path(audio_file.name).suffix
                else:
                    file_ext = '.wav'
                
                # Save the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    if hasattr(audio_file, 'read'):
                        tmp_file.write(audio_file.read())
                    else:
                        tmp_file.write(audio_file)
                    tmp_path = tmp_file.name
                
                # Convert to WAV if needed
                if file_ext.lower() != '.wav':
                    wav_path = tmp_path.replace(file_ext, '.wav')
                    y, sr = librosa.load(tmp_path, sr=16000)
                    sf.write(wav_path, y, sr)
                    os.unlink(tmp_path)
                    tmp_path = wav_path
                
                # Display audio player
                st.audio(tmp_path, format='audio/wav')
                
                # Analyze button
                if st.button("🔍 Analyze Audio"):
                    with st.spinner('Analyzing... This may take a moment...'):
                        try:
                            # Make prediction
                            result = st.session_state.detector.predict_file(tmp_path)
                            
                            # Display results
                            st.markdown("## 🎯 Analysis Results")
                            
                            # Result box
                            if result['prediction'] == "Alzheimer's":
                                result_class = "alzheimer"
                                emoji = "⚠️"
                                color = "red"
                            else:
                                result_class = "healthy"
                                emoji = "✅"
                                color = "green"
                            
                            st.markdown(f"""
                            <div class="result-box {result_class}">
                                <h2 style="color: {color};">{emoji} {result['prediction']} Detected</h2>
                                <h3>Confidence: <span style="color: {color};">{result['confidence']:.1f}%</span></h3>
                                
                                <div class="progress" style="height: 20px; background-color: #e0e0e0; border-radius: 10px; margin: 1rem 0;">
                                    <div style="width: {result['confidence']}%; height: 100%; background-color: {color}; border-radius: 10px;"></div>
                                </div>
                                
                                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                                    <span>Healthy</span>
                                    <span>Alzheimer's</span>
                                </div>
                                
                                <div class="progress" style="height: 20px; background: linear-gradient(to right, #4CAF50, #FFC107, #F44336); border-radius: 10px; margin: 0.5rem 0 1rem 0;">
                                    <div style="width: {result['probability_healthy']*100}%; height: 100%; background-color: rgba(0,0,0,0.1); border-right: 2px solid white;"></div>
                                </div>
                                
                                <div style="display: flex; justify-content: space-between;">
                                    <span>{result['probability_healthy']*100:.1f}%</span>
                                    <span>{result['probability_alzheimers']*100:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Interpretation
                            st.markdown("### 💡 Interpretation")
                            if result['prediction'] == "Alzheimer's":
                                st.warning("""
                                The analysis detected patterns consistent with Alzheimer's disease. 
                                
                                **Key indicators may include:**
                                - More frequent and longer pauses
                                - Slower speech rate
                                - Reduced pitch variation
                                - Irregular speech rhythm
                                
                                *Please consult a healthcare professional for a comprehensive evaluation.*
                                """)
                            else:
                                st.success("""
                                The analysis detected healthy speech patterns.
                                
                                **Key indicators:**
                                - Normal speech rate
                                - Good pitch variation
                                - Regular speech rhythm
                                - Few hesitations
                                
                                *This result does not guarantee the absence of cognitive issues. Regular check-ups are recommended.*
                                """)
                            
                            # Feature importance (top 5)
                            st.markdown("### 🔍 Key Features")
                            st.markdown("*Top 5 features that influenced this prediction:*")
                            
                            if 'top_features' in result:
                                for feat in result['top_features']:
                                    st.markdown(f"- **{feat['name']}**: {feat['value']:.4f} (importance: {feat['importance']:.2f})")
                            else:
                                st.info("Feature importance not available for this model.")
                            
                            # Disclaimer
                            st.markdown("---")
                            st.warning("""
                            **⚠️ Important Note:**  
                            This tool is for research and informational purposes only. It is not intended to diagnose, treat, or prevent any disease. Always consult with a qualified healthcare professional for medical advice.
                            """)
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            st.error("Please try a different audio file or check the console for details.")
                            
                            # Show debug info
                            with st.expander("Debug Information"):
                                st.write(f"File path: {tmp_path}")
                                st.write(f"File exists: {os.path.exists(tmp_path)}")
                                if os.path.exists(tmp_path):
                                    st.write(f"File size: {os.path.getsize(tmp_path)} bytes")
                                    try:
                                        y, sr = librosa.load(tmp_path, sr=None)
                                        st.write(f"Audio duration: {len(y)/sr:.2f} seconds")
                                        st.write(f"Sample rate: {sr} Hz")
                                    except Exception as load_err:
                                        st.write(f"Error loading audio: {load_err}")
                                st.write(f"Full error: {repr(e)}")
            except Exception as file_error:
                st.error(f"Error processing file: {str(file_error)}")
                st.error("Please ensure the file is a valid audio file.")
            finally:
                # Clean up temp file
                try:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass
    
    with col2:
        st.header("Quick Guide")
        st.markdown("""
        ### How to record good samples:
        
        1. Speak naturally for 30-60 seconds
        2. Use a quiet environment
        3. Keep a consistent distance from mic
        4. Avoid background noise
        
        ### Example prompts:
        - Describe your morning routine
        - Talk about your favorite holiday
        - Recite a story or memory
        
        ### What the model analyzes:
        - Speech rate and rhythm
        - Pitch variation
        - Voice quality
        - Pauses and hesitations
        - 97+ other features
        """)
        
        st.markdown("---")
        st.markdown("### Test Samples")
        st.markdown("*Click to analyze pre-loaded samples*")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔴 Alzheimer's", use_container_width=True):
                sample_path = Path("data/processed/alzheimer/alz_001.wav")
                if sample_path.exists():
                    with st.spinner('Analyzing Alzheimer\'s sample...'):
                        try:
                            result = st.session_state.detector.predict_file(sample_path)
                            st.success(f"Result: {result['prediction']}")
                            st.info(f"Confidence: {result['confidence']:.1f}%")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("Sample not found")
        
        with col_b:
            if st.button("🟢 Healthy", use_container_width=True):
                sample_path = Path("data/processed/healthy/healthy_001.wav")
                if sample_path.exists():
                    with st.spinner('Analyzing healthy sample...'):
                        try:
                            result = st.session_state.detector.predict_file(sample_path)
                            st.success(f"Result: {result['prediction']}")
                            st.info(f"Confidence: {result['confidence']:.1f}%")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("Sample not found")

if __name__ == "__main__":
    main()
