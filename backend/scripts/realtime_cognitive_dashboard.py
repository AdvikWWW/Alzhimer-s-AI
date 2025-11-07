#!/usr/bin/env python3
"""
Real-time Cognitive Assessment Dashboard
Interactive Streamlit app with live speech analysis and task-based evaluation
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
import sys
import tempfile
import librosa
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import cognitive assessment system
from cognitive_assessment_system import CognitiveAssessmentSystem, CognitiveTask

# Page configuration
st.set_page_config(
    page_title="Real-time Alzheimer's Cognitive Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
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
        margin-bottom: 2rem;
    }
    .task-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .score-excellent {
        color: #10b981;
        font-weight: bold;
    }
    .score-good {
        color: #3b82f6;
        font-weight: bold;
    }
    .score-moderate {
        color: #f59e0b;
        font-weight: bold;
    }
    .score-poor {
        color: #ef4444;
        font-weight: bold;
    }
    .real-time-indicator {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: transform 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'assessment_system' not in st.session_state:
    st.session_state.assessment_system = CognitiveAssessmentSystem()
    st.session_state.is_recording = False
    st.session_state.current_task = None
    st.session_state.task_results = []
    st.session_state.real_time_data = {
        'timestamps': [],
        'fluency': [],
        'coherence': [],
        'speech_rate': [],
        'pause_density': []
    }
    st.session_state.recording_thread = None
    st.session_state.start_time = None


def render_header():
    """Render application header"""
    st.markdown('<h1 class="main-header">🧠 Real-time Cognitive Assessment System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <p style="text-align: center; color: #666; font-size: 1.1rem;">
        Advanced Alzheimer's detection through interactive speech tasks and real-time analysis
        </p>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with task selection and information"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
        st.title("Cognitive Tasks")
        
        st.markdown("### 📋 Select Assessment Task")
        
        task_options = {
            "Picture Description": CognitiveTask.PICTURE_DESCRIPTION,
            "Story Recall": CognitiveTask.STORY_RECALL,
            "Verbal Fluency": CognitiveTask.VERBAL_FLUENCY,
            "Serial Subtraction": CognitiveTask.SERIAL_SUBTRACTION,
            "Free Speech": CognitiveTask.FREE_SPEECH
        }
        
        selected_task_name = st.selectbox(
            "Choose a task:",
            options=list(task_options.keys()),
            help="Select a cognitive assessment task to begin"
        )
        
        st.session_state.selected_task = task_options[selected_task_name]
        
        st.markdown("---")
        
        # Task description
        st.markdown("### 📖 Task Description")
        
        task_descriptions = {
            CognitiveTask.PICTURE_DESCRIPTION: """
            **Picture Description**
            - Describe a complex scene
            - Tests: Vocabulary, coherence, detail recognition
            - Duration: 60 seconds
            """,
            CognitiveTask.STORY_RECALL: """
            **Story Recall**
            - Listen and repeat a short story
            - Tests: Memory, accuracy, sequencing
            - Duration: 45 seconds
            """,
            CognitiveTask.VERBAL_FLUENCY: """
            **Verbal Fluency**
            - Name items in a category
            - Tests: Word retrieval, cognitive flexibility
            - Duration: 30 seconds
            """,
            CognitiveTask.SERIAL_SUBTRACTION: """
            **Serial Subtraction**
            - Count backwards by 7s from 100
            - Tests: Working memory, calculation
            - Duration: 60 seconds
            """,
            CognitiveTask.FREE_SPEECH: """
            **Free Speech**
            - Open conversation
            - Tests: Natural speech patterns
            - Duration: Variable
            """
        }
        
        st.info(task_descriptions[st.session_state.selected_task])
        
        st.markdown("---")
        
        # Scoring information
        st.markdown("### 📊 Scoring System")
        st.markdown("""
        **Cognitive Score (0-100)**
        - 🟢 **75-100**: Healthy
        - 🟡 **50-74**: Mild Impairment
        - 🔴 **0-49**: Possible Alzheimer's
        
        **Factors Analyzed:**
        - Speech fluency & rhythm
        - Semantic relevance
        - Pause patterns
        - Vocabulary diversity
        - Task completion
        """)
        
        st.markdown("---")
        st.warning("⚠️ **Disclaimer**: For research purposes only. Not for clinical diagnosis.")


def render_task_interface():
    """Render the main task interface"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Current Task")
        
        # Task prompt display
        if st.session_state.current_task:
            task_info = st.session_state.assessment_system.tasks[st.session_state.selected_task]
            
            st.markdown(f"""
            <div class="task-card">
                <h3>{st.session_state.selected_task.value.replace('_', ' ').title()}</h3>
                <p style="font-size: 1.1rem; margin: 1rem 0;">
                    {task_info.prompt_text}
                </p>
                <p style="font-size: 0.9rem; opacity: 0.9;">
                    Duration: {task_info.duration_seconds} seconds
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Select a task and click 'Start Recording' to begin")
    
    with col2:
        st.markdown("### 🎤 Recording Control")
        
        # Recording status
        if st.session_state.is_recording:
            st.markdown("""
            <div style="text-align: center;">
                <div class="real-time-indicator" style="font-size: 2rem; color: #ef4444;">
                    🔴 Recording...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show elapsed time
            if st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                st.metric("Elapsed Time", f"{elapsed:.1f}s")
        else:
            st.markdown("""
            <div style="text-align: center; font-size: 2rem; color: #6b7280;">
                ⭕ Ready
            </div>
            """, unsafe_allow_html=True)
        
        # Control buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🎙️ Start" if not st.session_state.is_recording else "⏸️ Pause",
                        key="record_btn",
                        use_container_width=True,
                        disabled=st.session_state.is_recording):
                start_recording()
        
        with col_btn2:
            if st.button("⏹️ Stop",
                        key="stop_btn",
                        use_container_width=True,
                        disabled=not st.session_state.is_recording):
                stop_recording()


def render_real_time_metrics():
    """Render real-time metrics dashboard"""
    st.markdown("### 📈 Real-time Analysis")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current metrics
    if st.session_state.is_recording:
        metrics = st.session_state.assessment_system.get_real_time_metrics()
    else:
        metrics = {
            'current_fluency': 0,
            'current_coherence': 0,
            'current_speech_rate': 0,
            'current_pause_density': 0
        }
    
    with col1:
        fluency_color = get_score_color(metrics['current_fluency'])
        st.markdown(f"""
        <div class="metric-card">
            <h4>Speech Fluency</h4>
            <p class="{fluency_color}" style="font-size: 2rem;">
                {metrics['current_fluency']:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        coherence_color = get_score_color(metrics['current_coherence'])
        st.markdown(f"""
        <div class="metric-card">
            <h4>Coherence</h4>
            <p class="{coherence_color}" style="font-size: 2rem;">
                {metrics['current_coherence']:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Speech rate (normal: 120-180 wpm)
        rate_color = "score-good" if 120 <= metrics['current_speech_rate'] <= 180 else "score-moderate"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Speech Rate</h4>
            <p class="{rate_color}" style="font-size: 2rem;">
                {metrics['current_speech_rate']:.0f} wpm
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pause_color = "score-good" if metrics['current_pause_density'] < 0.25 else "score-poor"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Pause Density</h4>
            <p class="{pause_color}" style="font-size: 2rem;">
                {metrics['current_pause_density']:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_real_time_graphs():
    """Render real-time visualization graphs"""
    st.markdown("### 📊 Live Monitoring")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Speech Fluency Trend", "Coherence Trend", 
                       "Speech Rate Variation", "Pause Pattern"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Get real-time data
    if st.session_state.is_recording and st.session_state.real_time_data['timestamps']:
        timestamps = st.session_state.real_time_data['timestamps']
        
        # Fluency trend
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=st.session_state.real_time_data['fluency'],
                mode='lines+markers',
                name='Fluency',
                line=dict(color='#667eea', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Coherence trend
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=st.session_state.real_time_data['coherence'],
                mode='lines+markers',
                name='Coherence',
                line=dict(color='#f5576c', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # Speech rate
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=st.session_state.real_time_data['speech_rate'],
                mode='lines+markers',
                name='Speech Rate',
                line=dict(color='#10b981', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # Pause density
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=st.session_state.real_time_data['pause_density'],
                name='Pause Density',
                marker_color='#f59e0b'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Score", row=1, col=2, range=[0, 1])
    fig.update_yaxes(title_text="WPM", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=2, range=[0, 0.5])
    
    st.plotly_chart(fig, use_container_width=True)


def render_results():
    """Render task results and scoring"""
    if st.session_state.task_results:
        st.markdown("### 🏆 Assessment Results")
        
        latest_result = st.session_state.task_results[-1]
        
        # Overall score display
        score = latest_result['overall_score']
        prediction = latest_result['prediction']
        
        score_color = get_score_category(score)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2>Cognitive Score</h2>
                <div style="font-size: 4rem; font-weight: bold; color: {score_color};">
                    {score:.0f}/100
                </div>
                <p style="font-size: 1.5rem; color: #666; margin-top: 1rem;">
                    {prediction.replace('_', ' ')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown("#### 📋 Detailed Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Speech Fluency", f"{latest_result['speech_fluency']:.1%}")
        
        with col2:
            st.metric("Semantic Relevance", f"{latest_result['semantic_relevance']:.1%}")
        
        with col3:
            st.metric("Coherence", f"{latest_result['coherence']:.1%}")
        
        with col4:
            st.metric("Lexical Diversity", f"{latest_result['lexical_diversity']:.1%}")
        
        # Trends
        if 'real_time_trends' in latest_result:
            st.markdown("#### 📈 Performance Trends")
            
            trends = latest_result['real_time_trends']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_icon = get_trend_icon(trends['fluency_trend'])
                st.metric("Fluency Trend", trends['fluency_trend'], delta=trend_icon)
            
            with col2:
                trend_icon = get_trend_icon(trends['coherence_trend'])
                st.metric("Coherence Trend", trends['coherence_trend'], delta=trend_icon)
            
            with col3:
                trend_icon = get_trend_icon(trends['speech_rate_trend'])
                st.metric("Speech Rate Trend", trends['speech_rate_trend'], delta=trend_icon)
        
        # Export results
        st.markdown("#### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download Report (JSON)", use_container_width=True):
                json_str = json.dumps(latest_result, indent=2, default=str)
                st.download_button(
                    label="Download",
                    data=json_str,
                    file_name=f"cognitive_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 View History", use_container_width=True):
                show_history()


def start_recording():
    """Start real-time recording"""
    st.session_state.is_recording = True
    st.session_state.start_time = time.time()
    st.session_state.current_task = st.session_state.selected_task
    
    # Reset real-time data
    st.session_state.real_time_data = {
        'timestamps': [],
        'fluency': [],
        'coherence': [],
        'speech_rate': [],
        'pause_density': []
    }
    
    # Start task
    task_info = st.session_state.assessment_system.start_task(st.session_state.selected_task)
    
    # Start recording
    st.session_state.assessment_system.start_real_time_recording()
    
    # Start update thread
    st.session_state.recording_thread = threading.Thread(target=update_real_time_data)
    st.session_state.recording_thread.start()
    
    st.rerun()


def stop_recording():
    """Stop recording and get results"""
    st.session_state.is_recording = False
    
    # Stop recording
    results = st.session_state.assessment_system.stop_real_time_recording()
    
    # Store results
    st.session_state.task_results.append(results)
    
    # Stop update thread
    if st.session_state.recording_thread:
        st.session_state.recording_thread.join(timeout=1)
    
    st.rerun()


def update_real_time_data():
    """Update real-time data in background"""
    while st.session_state.is_recording:
        try:
            # Get current metrics
            metrics = st.session_state.assessment_system.get_real_time_metrics()
            
            # Update data
            current_time = time.time() - st.session_state.start_time
            st.session_state.real_time_data['timestamps'].append(current_time)
            st.session_state.real_time_data['fluency'].append(metrics['current_fluency'])
            st.session_state.real_time_data['coherence'].append(metrics['current_coherence'])
            st.session_state.real_time_data['speech_rate'].append(metrics['current_speech_rate'])
            st.session_state.real_time_data['pause_density'].append(metrics['current_pause_density'])
            
            # Keep only last 30 seconds of data
            if len(st.session_state.real_time_data['timestamps']) > 30:
                for key in st.session_state.real_time_data:
                    st.session_state.real_time_data[key] = st.session_state.real_time_data[key][-30:]
            
            time.sleep(1)  # Update every second
            
        except Exception as e:
            print(f"Error updating real-time data: {e}")
            break


def get_score_color(score):
    """Get color class based on score"""
    if score >= 0.75:
        return "score-excellent"
    elif score >= 0.5:
        return "score-good"
    elif score >= 0.25:
        return "score-moderate"
    else:
        return "score-poor"


def get_score_category(score):
    """Get color for score category"""
    if score >= 75:
        return "#10b981"  # Green
    elif score >= 50:
        return "#f59e0b"  # Yellow
    else:
        return "#ef4444"  # Red


def get_trend_icon(trend):
    """Get icon for trend"""
    if trend == "improving":
        return "↑"
    elif trend == "declining":
        return "↓"
    else:
        return "→"


def show_history():
    """Show assessment history"""
    if st.session_state.task_results:
        df = pd.DataFrame(st.session_state.task_results)
        st.dataframe(df, use_container_width=True)


def main():
    """Main application"""
    render_header()
    render_sidebar()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Assessment", "📊 Real-time Monitor", "📈 Results"])
    
    with tab1:
        render_task_interface()
        st.markdown("---")
        render_real_time_metrics()
    
    with tab2:
        if st.session_state.is_recording:
            # Auto-refresh for real-time updates
            st.empty()
            time.sleep(1)
            st.rerun()
        
        render_real_time_graphs()
    
    with tab3:
        render_results()


if __name__ == "__main__":
    main()
