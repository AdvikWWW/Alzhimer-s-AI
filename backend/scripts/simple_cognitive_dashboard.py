#!/usr/bin/env python3
"""
Enhanced Real-time Cognitive Assessment Dashboard
With file upload, recording save, and advanced analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sounddevice as sd
import queue
import threading
import time
from datetime import datetime
import json
from pathlib import Path
import tempfile
import librosa
import soundfile as sf
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import advanced analyzer
try:
    from scripts.enhanced_word_level_analyzer import WordLevelAnalyzer
    ADVANCED_ANALYSIS_AVAILABLE = True
except:
    ADVANCED_ANALYSIS_AVAILABLE = False
    print("⚠️ Advanced analysis not available - using basic features only")

# Page configuration
st.set_page_config(
    page_title="Real-time Alzheimer's Cognitive Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .task-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class CognitiveTask(Enum):
    """Types of cognitive assessment tasks"""
    PICTURE_DESCRIPTION = "picture_description"
    STORY_RECALL = "story_recall"
    VERBAL_FLUENCY = "verbal_fluency"
    SERIAL_SUBTRACTION = "serial_subtraction"
    FREE_SPEECH = "free_speech"


# Create recordings directory
RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Initialize session state
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
    st.session_state.audio_data = []
    st.session_state.sample_rate = 16000
    st.session_state.start_time = None
    st.session_state.results = None
    st.session_state.saved_audio_path = None
    st.session_state.uploaded_file_path = None
    st.session_state.real_time_metrics = {
        'timestamps': [],
        'word_count': [],
        'pause_count': [],
        'speech_rate': []
    }
    # Initialize advanced analyzer if available
    if ADVANCED_ANALYSIS_AVAILABLE:
        try:
            st.session_state.advanced_analyzer = WordLevelAnalyzer(use_gpu=False)
        except:
            st.session_state.advanced_analyzer = None
    else:
        st.session_state.advanced_analyzer = None


def get_task_info(task_type):
    """Get task information"""
    tasks = {
        CognitiveTask.PICTURE_DESCRIPTION: {
            'name': 'Picture Description',
            'prompt': 'Please describe what you see happening in this picture. Include details about the people, objects, and actions.',
            'duration': 60,
            'icon': '🖼️'
        },
        CognitiveTask.STORY_RECALL: {
            'name': 'Story Recall',
            'prompt': "I'll tell you a short story. Please listen carefully and then repeat it back to me:\n\n'Sarah went to the grocery store. She bought apples, bread, and milk. On her way home, she met her neighbor Tom. They talked about the weather.'\n\nNow please repeat the story.",
            'duration': 45,
            'icon': '📖'
        },
        CognitiveTask.VERBAL_FLUENCY: {
            'name': 'Verbal Fluency',
            'prompt': 'Name as many animals as you can in the next 30 seconds. Begin now.',
            'duration': 30,
            'icon': '🗣️'
        },
        CognitiveTask.SERIAL_SUBTRACTION: {
            'name': 'Serial Subtraction',
            'prompt': 'Start from 100 and subtract 7 each time. Continue as far as you can. Begin now.',
            'duration': 60,
            'icon': '🔢'
        },
        CognitiveTask.FREE_SPEECH: {
            'name': 'Free Speech',
            'prompt': 'Tell me about your day or anything you would like to talk about.',
            'duration': 60,
            'icon': '💬'
        }
    }
    return tasks.get(task_type, tasks[CognitiveTask.FREE_SPEECH])


def audio_callback(indata, frames, time_info, status):
    """Callback for audio recording"""
    if status:
        print(f"Audio status: {status}")
    if st.session_state.is_recording:
        st.session_state.audio_data.append(indata.copy())


def start_recording():
    """Start audio recording"""
    st.session_state.is_recording = True
    st.session_state.audio_data = []
    st.session_state.start_time = time.time()
    st.session_state.results = None
    
    # Start audio stream
    st.session_state.stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=st.session_state.sample_rate
    )
    st.session_state.stream.start()


def stop_recording():
    """Stop recording and analyze"""
    st.session_state.is_recording = False
    
    if hasattr(st.session_state, 'stream'):
        st.session_state.stream.stop()
        st.session_state.stream.close()
    
    # Analyze audio
    if st.session_state.audio_data:
        # Save recording first
        audio = np.concatenate(st.session_state.audio_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        task_name = st.session_state.selected_task.value if hasattr(st.session_state, 'selected_task') else 'recording'
        filename = f"{task_name}_{timestamp}.wav"
        audio_path = RECORDINGS_DIR / filename
        
        sf.write(str(audio_path), audio, st.session_state.sample_rate)
        st.session_state.saved_audio_path = str(audio_path)
        
        # Now analyze
        analyze_recording(audio_path)


def analyze_recording(audio_path):
    """Analyze the recorded or uploaded audio with advanced features"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        # Basic analysis
        frame_length = 2048
        hop_length = 512
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.mean(energy) * 0.5
        speech_frames = energy > threshold
        
        # Calculate basic metrics
        speech_ratio = np.sum(speech_frames) / len(speech_frames)
        estimated_words = int((duration * speech_ratio) * 2.5)
        pause_transitions = np.diff(speech_frames.astype(int))
        pause_count = np.sum(pause_transitions == -1)
        speech_rate = (estimated_words / duration) * 60 if duration > 0 else 0
        
        # Extract advanced features if available
        advanced_features = {}
        if st.session_state.advanced_analyzer is not None:
            try:
                with st.spinner("🔬 Running advanced analysis..."):
                    # Extract comprehensive acoustic features
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
                    zcr = librosa.feature.zero_crossing_rate(audio)
                    
                    advanced_features = {
                        'mfcc_mean': float(np.mean(mfccs)),
                        'mfcc_std': float(np.std(mfccs)),
                        'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                        'zcr_mean': float(np.mean(zcr)),
                        'energy_mean': float(np.mean(energy)),
                        'energy_std': float(np.std(energy))
                    }
                    
                    # Try to extract pitch
                    try:
                        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
                        pitch_values = []
                        for t in range(pitches.shape[1]):
                            index = magnitudes[:, t].argmax()
                            pitch = pitches[index, t]
                            if pitch > 0:
                                pitch_values.append(pitch)
                        if pitch_values:
                            advanced_features['pitch_mean'] = float(np.mean(pitch_values))
                            advanced_features['pitch_std'] = float(np.std(pitch_values))
                    except:
                        pass
            except Exception as e:
                st.warning(f"⚠️ Advanced analysis partially failed: {str(e)}")
        
        # Calculate enhanced cognitive score
        score = 50  # Base score
        
        # Adjust based on metrics
        if speech_ratio > 0.6:
            score += 15
        elif speech_ratio < 0.3:
            score -= 15
        
        if 100 < speech_rate < 180:
            score += 15
        elif speech_rate < 80:
            score -= 10
        
        if estimated_words > 20:
            score += 10
        
        if pause_count < duration * 2:
            score += 10
        else:
            score -= 10
        
        # Adjust with advanced features if available
        if advanced_features:
            # Spectral features indicate voice quality
            if advanced_features.get('spectral_centroid_mean', 0) > 1000:
                score += 5
            if advanced_features.get('zcr_mean', 0) < 0.1:
                score += 5
        
        score = max(0, min(100, score))
        
        # Determine prediction
        if score >= 75:
            prediction = "Healthy"
            risk_level = "Low Risk"
        elif score >= 50:
            prediction = "Mild Cognitive Impairment"
            risk_level = "Moderate Risk"
        else:
            prediction = "Possible Alzheimer's"
            risk_level = "High Risk"
        
        # Store results
        st.session_state.results = {
            'audio_path': audio_path,
            'duration': duration,
            'speech_ratio': speech_ratio,
            'estimated_words': estimated_words,
            'pause_count': pause_count,
            'speech_rate': speech_rate,
            'cognitive_score': score,
            'prediction': prediction,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'advanced_features': advanced_features,
            'analysis_type': 'advanced' if advanced_features else 'basic'
        }
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.session_state.results = None


def render_header():
    """Render header"""
    st.markdown('<h1 class="main-header">🧠 Real-time Cognitive Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 1.1rem;">
    Advanced Alzheimer's detection through interactive speech tasks
    </p>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
        st.title("Cognitive Tasks")
        
        # File upload section
        st.markdown("### 📤 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Upload WAV or MP3 file",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload a pre-recorded audio file for analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_extension = uploaded_file.name.split('.')[-1]
            upload_path = RECORDINGS_DIR / f"uploaded_{timestamp}.{file_extension}"
            
            with open(upload_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.uploaded_file_path = str(upload_path)
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            if st.button("🔬 Analyze Uploaded File", use_container_width=True):
                with st.spinner("Analyzing..."):
                    analyze_recording(st.session_state.uploaded_file_path)
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📋 Select Assessment Task")
        
        task_options = {
            "🖼️ Picture Description": CognitiveTask.PICTURE_DESCRIPTION,
            "📖 Story Recall": CognitiveTask.STORY_RECALL,
            "🗣️ Verbal Fluency": CognitiveTask.VERBAL_FLUENCY,
            "🔢 Serial Subtraction": CognitiveTask.SERIAL_SUBTRACTION,
            "💬 Free Speech": CognitiveTask.FREE_SPEECH
        }
        
        selected = st.selectbox("Choose a task:", list(task_options.keys()))
        st.session_state.selected_task = task_options[selected]
        
        st.markdown("---")
        
        # Task info
        task_info = get_task_info(st.session_state.selected_task)
        st.markdown(f"### {task_info['icon']} {task_info['name']}")
        st.info(f"**Duration:** {task_info['duration']} seconds\n\n{task_info['prompt'][:100]}...")
        
        st.markdown("---")
        st.markdown("### 📊 Scoring")
        st.markdown("""
        - 🟢 **75-100**: Healthy
        - 🟡 **50-74**: MCI
        - 🔴 **0-49**: Possible Alzheimer's
        """)
        
        # Show analysis status
        if ADVANCED_ANALYSIS_AVAILABLE and st.session_state.advanced_analyzer:
            st.success("✅ Advanced analysis enabled")
        else:
            st.info("ℹ️ Basic analysis mode")
        
        st.warning("⚠️ For research purposes only")


def render_main_interface():
    """Render main interface"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Current Task")
        
        task_info = get_task_info(st.session_state.selected_task)
        
        st.markdown(f"""
        <div class="task-card">
            <h3>{task_info['icon']} {task_info['name']}</h3>
            <p style="font-size: 1.1rem; margin: 1rem 0;">
                {task_info['prompt']}
            </p>
            <p style="font-size: 0.9rem; opacity: 0.9;">
                Duration: {task_info['duration']} seconds
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎤 Recording Control")
        
        if st.session_state.is_recording:
            st.markdown('<div style="text-align: center; font-size: 2rem; color: #ef4444;">🔴 Recording...</div>', unsafe_allow_html=True)
            if st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                st.metric("Elapsed Time", f"{elapsed:.1f}s")
        else:
            st.markdown('<div style="text-align: center; font-size: 2rem; color: #6b7280;">⭕ Ready</div>', unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🎙️ Start", disabled=st.session_state.is_recording, use_container_width=True):
                start_recording()
                st.rerun()
        
        with col_btn2:
            if st.button("⏹️ Stop", disabled=not st.session_state.is_recording, use_container_width=True):
                stop_recording()
                st.rerun()


def render_results():
    """Render results"""
    if st.session_state.results:
        st.markdown("### 🏆 Assessment Results")
        
        results = st.session_state.results
        score = results['cognitive_score']
        
        # Score display
        if score >= 75:
            color = "#10b981"
        elif score >= 50:
            color = "#f59e0b"
        else:
            color = "#ef4444"
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2>Cognitive Score</h2>
                <div style="font-size: 4rem; font-weight: bold; color: {color};">
                    {score:.0f}/100
                </div>
                <p style="font-size: 1.5rem; color: #666; margin-top: 1rem;">
                    {results['prediction']}
                </p>
                <p style="font-size: 1.2rem; color: #999;">
                    {results['risk_level']}
                </p>
                <p style="font-size: 0.9rem; color: #aaa; margin-top: 0.5rem;">
                    Analysis: {results.get('analysis_type', 'basic').upper()}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown("#### 📋 Detailed Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{results['duration']:.1f}s")
        
        with col2:
            st.metric("Estimated Words", results['estimated_words'])
        
        with col3:
            st.metric("Speech Rate", f"{results['speech_rate']:.0f} wpm")
        
        with col4:
            st.metric("Speech Ratio", f"{results['speech_ratio']:.1%}")
        
        # Advanced features if available
        if results.get('advanced_features'):
            st.markdown("#### 🔬 Advanced Acoustic Features")
            adv_features = results['advanced_features']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'mfcc_mean' in adv_features:
                    st.metric("MFCC Mean", f"{adv_features['mfcc_mean']:.2f}")
            
            with col2:
                if 'spectral_centroid_mean' in adv_features:
                    st.metric("Spectral Centroid", f"{adv_features['spectral_centroid_mean']:.0f} Hz")
            
            with col3:
                if 'pitch_mean' in adv_features:
                    st.metric("Pitch Mean", f"{adv_features['pitch_mean']:.0f} Hz")
            
            with col4:
                if 'energy_mean' in adv_features:
                    st.metric("Energy Mean", f"{adv_features['energy_mean']:.3f}")
        
        # Audio file info
        if results.get('audio_path'):
            st.markdown("#### 🎵 Recorded Audio")
            audio_path = Path(results['audio_path'])
            if audio_path.exists():
                st.success(f"✅ Recording saved: `{audio_path.name}`")
                st.info(f"📁 Location: `{audio_path.parent}`")
                
                # Play audio
                with open(audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
                
                # Download audio
                st.download_button(
                    label="⬇️ Download Audio File",
                    data=audio_bytes,
                    file_name=audio_path.name,
                    mime="audio/wav"
                )
        
        # Export results
        st.markdown("#### 💾 Export Results")
        json_str = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_str,
            file_name=f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Main app"""
    render_header()
    render_sidebar()
    
    st.markdown("---")
    
    render_main_interface()
    
    st.markdown("---")
    
    render_results()
    
    # Auto-refresh during recording
    if st.session_state.is_recording:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
