import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Mic, User, Clock, Shield, AlertTriangle } from 'lucide-react'
import { ParticipantForm } from '../components/ParticipantForm'

export const HomePage: React.FC = () => {
  const navigate = useNavigate()
  const [showForm, setShowForm] = useState(false)

  const handleStartSession = (participantData: any) => {
    // Create new session and navigate
    const sessionId = `session_${Date.now()}`
    navigate(`/session/${sessionId}`, { state: { participantData } })
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Voice Biomarker Assessment Platform
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
          Advanced voice analysis for Alzheimer's disease research using clinically-validated biomarkers
          and machine learning models trained on established datasets.
        </p>
        
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 max-w-2xl mx-auto mb-8">
          <div className="flex items-center justify-center mb-2">
            <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
            <span className="font-semibold text-red-800">Research Use Only</span>
          </div>
          <p className="text-red-700 text-sm">
            This platform is designed for research purposes only. Results should not be used for clinical diagnosis
            and must be interpreted by qualified healthcare professionals.
          </p>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-3 gap-6 mb-12">
        <div className="card text-center">
          <Mic className="h-12 w-12 text-medical-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Structured Recording</h3>
          <p className="text-gray-600">
            Guided tasks including narrative, picture description, semantic fluency, and reading passages
          </p>
        </div>
        
        <div className="card text-center">
          <User className="h-12 w-12 text-medical-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Clinical Biomarkers</h3>
          <p className="text-gray-600">
            Extract disfluencies, acoustic features, lexical-semantic markers, and speech timing patterns
          </p>
        </div>
        
        <div className="card text-center">
          <Clock className="h-12 w-12 text-medical-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Real-time Analysis</h3>
          <p className="text-gray-600">
            Advanced ML pipeline with ensemble models and calibrated uncertainty measures
          </p>
        </div>
      </div>

      {/* Recording Tasks Overview */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-6">Assessment Tasks</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-3">Cognitive Tasks</h3>
            <ul className="space-y-2 text-gray-600">
              <li>• Free narrative description</li>
              <li>• Picture description (Cookie Theft)</li>
              <li>• Semantic fluency (animal naming)</li>
              <li>• Confrontation naming</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-3">Speech Tasks</h3>
            <ul className="space-y-2 text-gray-600">
              <li>• Reading passage (Rainbow Passage)</li>
              <li>• Sentence repetition</li>
              <li>• Conversational prompts</li>
              <li>• Microphone calibration</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Start Session */}
      <div className="text-center">
        {!showForm ? (
          <button
            onClick={() => setShowForm(true)}
            className="btn-medical text-lg px-8 py-4"
          >
            <Mic className="h-5 w-5 mr-2 inline" />
            Start New Assessment
          </button>
        ) : (
          <ParticipantForm onSubmit={handleStartSession} />
        )}
      </div>

      {/* Technical Details */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-6">Technical Approach</h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="font-semibold mb-3 text-medical-700">Audio Processing Pipeline</h3>
            <ul className="space-y-1 text-sm text-gray-600">
              <li>• Voice Activity Detection (VAD)</li>
              <li>• Speaker diarization with PyAnnote</li>
              <li>• Multi-engine ASR (WhisperX primary)</li>
              <li>• Forced alignment for precise timing</li>
              <li>• Acoustic feature extraction (Librosa/Parselmouth)</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-3 text-medical-700">Biomarker Extraction</h3>
            <ul className="space-y-1 text-sm text-gray-600">
              <li>• Disfluency detection (pauses, repetitions)</li>
              <li>• Acoustic features (pitch, jitter, shimmer, HNR)</li>
              <li>• Lexical diversity and semantic coherence</li>
              <li>• Speech timing and prosody analysis</li>
              <li>• Voice quality assessment</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h3 className="font-semibold mb-3 text-medical-700">Research Foundation</h3>
          <div className="text-sm text-gray-600 space-y-1">
            <p>• <strong>López-de Ipiña et al. (2013):</strong> Speech disfluencies in AD detection</p>
            <p>• <strong>Saeedi et al. (2024):</strong> Acoustic features and voice quality markers</p>
            <p>• <strong>Favaro et al. (2023):</strong> Lexical-semantic coherence analysis</p>
            <p>• <strong>Yang et al. (2022):</strong> Speech timing and fluency measures</p>
          </div>
        </div>
      </div>
    </div>
  )
}
