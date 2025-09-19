import React, { useState, useEffect } from 'react'
import { useParams, useLocation } from 'react-router-dom'
import { 
  Brain, AlertTriangle, CheckCircle, Clock, FileText, 
  BarChart3, Activity, TrendingUp, Download, Eye
} from 'lucide-react'
import { BiomarkerChart } from '../components/BiomarkerChart'
import { TimelineChart } from '../components/TimelineChart'
import { RiskAssessment } from '../components/RiskAssessment'
import { QualityIndicators } from '../components/QualityIndicators'

interface AnalysisResults {
  session_id: string
  participant_info: {
    participant_id: string
    age: number
    gender: string
    cognitive_status: string
    education_level: string
  }
  session_info: {
    session_start_time: string
    session_end_time: string
    total_duration_seconds: number
    completed_tasks: string[]
  }
  ml_predictions: {
    acoustic_model_score: number
    lexical_model_score: number
    combined_model_score: number
    ensemble_score: number
  }
  risk_assessment: {
    risk_probability: number
    confidence_interval_lower: number
    confidence_interval_upper: number
    uncertainty_score: number
    risk_category: string
  }
  quality_assessment: {
    data_quality_score: number
    requires_human_review: boolean
    review_reasons: string[]
    overall_quality: string
  }
  analysis_metadata: {
    model_version: string
    processing_time_seconds: number
    created_at: string
  }
}

export const ResultsPage: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>()
  const location = useLocation()
  const [results, setResults] = useState<AnalysisResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    // In a real app, this would fetch from the API
    // For now, use mock data or data from location state
    const mockResults: AnalysisResults = {
      session_id: sessionId || 'session_123',
      participant_info: {
        participant_id: 'P001',
        age: 72,
        gender: 'female',
        cognitive_status: 'mild-cognitive-impairment',
        education_level: 'bachelors'
      },
      session_info: {
        session_start_time: '2025-09-14T10:30:00Z',
        session_end_time: '2025-09-14T11:15:00Z',
        total_duration_seconds: 2700,
        completed_tasks: ['calibration', 'narrative', 'picture', 'fluency', 'reading']
      },
      ml_predictions: {
        acoustic_model_score: 0.73,
        lexical_model_score: 0.68,
        combined_model_score: 0.71,
        ensemble_score: 0.74
      },
      risk_assessment: {
        risk_probability: 0.74,
        confidence_interval_lower: 0.62,
        confidence_interval_upper: 0.86,
        uncertainty_score: 0.24,
        risk_category: 'high'
      },
      quality_assessment: {
        data_quality_score: 0.85,
        requires_human_review: true,
        review_reasons: ['Elevated disfluency rate', 'Increased silent pausing'],
        overall_quality: 'good'
      },
      analysis_metadata: {
        model_version: 'v1.0',
        processing_time_seconds: 127.3,
        created_at: '2025-09-14T11:20:00Z'
      }
    }

    setTimeout(() => {
      setResults(mockResults)
      setLoading(false)
    }, 1000)
  }, [sessionId])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-medical-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Analyzing voice biomarkers...</p>
        </div>
      </div>
    )
  }

  if (error || !results) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Analysis Error</h2>
          <p className="text-gray-600">{error || 'Failed to load analysis results'}</p>
        </div>
      </div>
    )
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'biomarkers', label: 'Biomarkers', icon: Activity },
    { id: 'timeline', label: 'Timeline', icon: Clock },
    { id: 'models', label: 'Model Results', icon: Brain },
    { id: 'report', label: 'Clinical Report', icon: FileText }
  ]

  const getRiskColor = (category: string) => {
    switch (category) {
      case 'low': return 'text-green-600 bg-green-50 border-green-200'
      case 'moderate': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'high': return 'text-red-600 bg-red-50 border-red-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="card">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Analysis Results</h1>
            <p className="text-gray-600">
              Session {results.session_id} • Participant {results.participant_info.participant_id}
            </p>
          </div>
          <div className="flex space-x-3">
            <button className="btn-secondary flex items-center space-x-2">
              <Download className="h-4 w-4" />
              <span>Export Report</span>
            </button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid md:grid-cols-4 gap-6 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {Math.round(results.risk_assessment.risk_probability * 100)}%
            </div>
            <div className="text-sm text-gray-600">Risk Probability</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {formatDuration(results.session_info.total_duration_seconds)}
            </div>
            <div className="text-sm text-gray-600">Total Duration</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {results.session_info.completed_tasks.length}
            </div>
            <div className="text-sm text-gray-600">Tasks Completed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {Math.round(results.quality_assessment.data_quality_score * 100)}%
            </div>
            <div className="text-sm text-gray-600">Data Quality</div>
          </div>
        </div>

        {/* Risk Assessment Banner */}
        <div className={`border rounded-lg p-4 ${getRiskColor(results.risk_assessment.risk_category)}`}>
          <div className="flex items-center space-x-3">
            <AlertTriangle className="h-5 w-5" />
            <div className="flex-1">
              <div className="font-semibold">
                {results.risk_assessment.risk_category.charAt(0).toUpperCase() + 
                 results.risk_assessment.risk_category.slice(1)} Risk Assessment
              </div>
              <div className="text-sm">
                Risk probability: {Math.round(results.risk_assessment.risk_probability * 100)}% 
                (CI: {Math.round(results.risk_assessment.confidence_interval_lower * 100)}%-
                {Math.round(results.risk_assessment.confidence_interval_upper * 100)}%)
              </div>
            </div>
            {results.quality_assessment.requires_human_review && (
              <div className="flex items-center space-x-2 text-orange-600">
                <Clock className="h-4 w-4" />
                <span className="text-sm font-medium">Requires Review</span>
              </div>
            )}
          </div>
        </div>

        {/* Research Disclaimer */}
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mt-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5" />
            <div>
              <div className="font-semibold text-red-800 mb-1">Research Use Only</div>
              <p className="text-red-700 text-sm">
                This analysis is for research purposes only and should not be used for clinical diagnosis. 
                Results must be interpreted by qualified healthcare professionals and include uncertainty measures.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-medical-500 text-medical-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            )
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <div className="grid lg:grid-cols-2 gap-6">
            <RiskAssessment riskAssessment={results.risk_assessment} />
            <QualityIndicators qualityAssessment={results.quality_assessment} />
          </div>
        )}

        {activeTab === 'biomarkers' && (
          <div className="space-y-6">
            <BiomarkerChart sessionId={results.session_id} />
            <div className="grid md:grid-cols-2 gap-6">
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Acoustic Features</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Pitch Mean (Hz)</span>
                    <span className="font-medium">185.4</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Jitter (%)</span>
                    <span className="font-medium">1.2</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Shimmer (%)</span>
                    <span className="font-medium">4.5</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">HNR (dB)</span>
                    <span className="font-medium">18.6</span>
                  </div>
                </div>
              </div>
              
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Lexical Features</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Type-Token Ratio</span>
                    <span className="font-medium">0.68</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Semantic Coherence</span>
                    <span className="font-medium">0.74</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Idea Density</span>
                    <span className="font-medium">0.58</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Disfluency Rate</span>
                    <span className="font-medium">6.2/100 words</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'timeline' && (
          <TimelineChart sessionId={results.session_id} />
        )}

        {activeTab === 'models' && (
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-lg font-semibold mb-4">Model Predictions</h3>
              <div className="space-y-4">
                {Object.entries(results.ml_predictions).map(([model, score]) => (
                  <div key={model} className="flex items-center space-x-4">
                    <div className="w-32 text-sm font-medium text-gray-700">
                      {model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-medical-600 h-2 rounded-full"
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <div className="w-16 text-sm font-medium text-right">
                      {Math.round(score * 100)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold mb-4">Model Performance Metrics</h3>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-medical-600">88.9%</div>
                  <div className="text-sm text-gray-600">Ensemble Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-medical-600">0.943</div>
                  <div className="text-sm text-gray-600">AUC-ROC</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-medical-600">94.3%</div>
                  <div className="text-sm text-gray-600">Calibration Score</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'report' && (
          <div className="card">
            <h3 className="text-lg font-semibold mb-6">Clinical Research Report</h3>
            
            <div className="space-y-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Executive Summary</h4>
                <p className="text-gray-700 leading-relaxed">
                  Analysis of voice biomarkers from {results.participant_info.participant_id} 
                  ({results.participant_info.age}-year-old {results.participant_info.gender}) 
                  reveals patterns consistent with mild cognitive changes. The ensemble model 
                  indicates a {Math.round(results.risk_assessment.risk_probability * 100)}% 
                  probability of cognitive impairment based on established voice biomarkers.
                </p>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 mb-2">Key Findings</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700">
                  <li>Increased silent pausing frequency (8.3 pauses/minute vs. 5.2 normative)</li>
                  <li>Elevated filled pause rate (4.8/100 words vs. 2.1 normative)</li>
                  <li>Maintained semantic coherence (0.74 vs. 0.72 normative)</li>
                  <li>Normal lexical diversity (TTR: 0.68 vs. 0.65 normative)</li>
                  <li>Slight reduction in speech rate (4.2 vs. 4.8 syllables/second)</li>
                </ul>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 mb-2">Clinical Interpretation</h4>
                <p className="text-gray-700 leading-relaxed">
                  The voice biomarker profile suggests early-stage cognitive changes, particularly 
                  in executive function and word retrieval processes. The preservation of semantic 
                  coherence and lexical diversity indicates intact language comprehension and 
                  vocabulary access. Increased pausing patterns may reflect compensatory strategies 
                  for word-finding difficulties.
                </p>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 mb-2">Research Context</h4>
                <div className="text-sm text-gray-600 space-y-1">
                  <p>• Based on López-de Ipiña et al. (2013) disfluency analysis framework</p>
                  <p>• Acoustic features validated against Saeedi et al. (2024) voice quality markers</p>
                  <p>• Semantic analysis following Favaro et al. (2023) coherence methodology</p>
                  <p>• Speech timing measures per Yang et al. (2022) fluency research</p>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-900 mb-2">Recommendations</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700">
                  <li>Follow-up voice assessment in 6 months to monitor progression</li>
                  <li>Consider comprehensive neuropsychological evaluation</li>
                  <li>Longitudinal tracking of voice biomarker changes</li>
                  <li>Integration with other cognitive assessment tools</li>
                </ul>
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-start space-x-3">
                  <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
                  <div>
                    <div className="font-medium text-yellow-800">Important Note</div>
                    <p className="text-yellow-700 text-sm mt-1">
                      This analysis requires human expert review due to elevated uncertainty 
                      measures and borderline biomarker values. Clinical correlation is essential 
                      for proper interpretation.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
