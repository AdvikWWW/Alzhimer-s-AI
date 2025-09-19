import React from 'react'
import { CheckCircle, AlertTriangle, XCircle, Clock, Eye } from 'lucide-react'

interface QualityIndicatorsProps {
  qualityAssessment: {
    data_quality_score: number
    requires_human_review: boolean
    review_reasons: string[]
    overall_quality: string
  }
}

export const QualityIndicators: React.FC<QualityIndicatorsProps> = ({ qualityAssessment }) => {
  const getQualityColor = (quality: string) => {
    switch (quality) {
      case 'excellent': return 'text-green-600 bg-green-50 border-green-200'
      case 'good': return 'text-blue-600 bg-blue-50 border-blue-200'
      case 'fair': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'poor': return 'text-red-600 bg-red-50 border-red-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getQualityIcon = (quality: string) => {
    switch (quality) {
      case 'excellent': return CheckCircle
      case 'good': return CheckCircle
      case 'fair': return AlertTriangle
      case 'poor': return XCircle
      default: return AlertTriangle
    }
  }

  const QualityIcon = getQualityIcon(qualityAssessment.overall_quality)

  return (
    <div className="card">
      <div className="flex items-center space-x-3 mb-6">
        <QualityIcon className={`h-6 w-6 ${getQualityColor(qualityAssessment.overall_quality).split(' ')[0]}`} />
        <h3 className="text-lg font-semibold">Data Quality Assessment</h3>
      </div>

      {/* Overall Quality Score */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Overall Data Quality</span>
          <span className="text-lg font-bold">
            {Math.round(qualityAssessment.data_quality_score * 100)}%
          </span>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div 
            className={`h-3 rounded-full transition-all duration-300 ${
              qualityAssessment.data_quality_score >= 0.8 ? 'bg-green-500' :
              qualityAssessment.data_quality_score >= 0.6 ? 'bg-blue-500' :
              qualityAssessment.data_quality_score >= 0.4 ? 'bg-yellow-500' :
              'bg-red-500'
            }`}
            style={{ width: `${qualityAssessment.data_quality_score * 100}%` }}
          />
        </div>
        
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Poor</span>
          <span>Fair</span>
          <span>Good</span>
          <span>Excellent</span>
        </div>
      </div>

      {/* Quality Status */}
      <div className={`border rounded-lg p-4 mb-6 ${getQualityColor(qualityAssessment.overall_quality)}`}>
        <div className="flex items-center space-x-2 mb-2">
          <QualityIcon className="h-4 w-4" />
          <span className="font-medium">
            {qualityAssessment.overall_quality.charAt(0).toUpperCase() + qualityAssessment.overall_quality.slice(1)} Quality
          </span>
        </div>
        <p className="text-sm">
          {qualityAssessment.overall_quality === 'excellent' && 
            'Excellent data quality with complete feature extraction and high confidence scores.'}
          {qualityAssessment.overall_quality === 'good' && 
            'Good data quality with most features successfully extracted and reliable analysis.'}
          {qualityAssessment.overall_quality === 'fair' && 
            'Fair data quality with some missing features or reduced confidence in analysis.'}
          {qualityAssessment.overall_quality === 'poor' && 
            'Poor data quality with significant missing features or low confidence scores.'}
        </p>
      </div>

      {/* Human Review Status */}
      {qualityAssessment.requires_human_review && (
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2 mb-2">
            <Eye className="h-4 w-4 text-orange-600" />
            <span className="font-medium text-orange-800">Human Review Required</span>
          </div>
          <p className="text-sm text-orange-700 mb-3">
            This analysis has been flagged for expert review due to quality concerns or borderline results.
          </p>
          
          {qualityAssessment.review_reasons.length > 0 && (
            <div>
              <div className="text-sm font-medium text-orange-800 mb-1">Review Reasons:</div>
              <ul className="text-sm text-orange-700 space-y-1">
                {qualityAssessment.review_reasons.map((reason, index) => (
                  <li key={index} className="flex items-start space-x-1">
                    <span className="text-orange-500 mt-1">•</span>
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Quality Metrics Breakdown */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-900">Quality Metrics</h4>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-sm text-gray-600">Audio Quality</div>
            <div className="text-lg font-semibold text-gray-900">92%</div>
            <div className="text-xs text-gray-500">Clear recording, minimal noise</div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-sm text-gray-600">Transcription</div>
            <div className="text-lg font-semibold text-gray-900">87%</div>
            <div className="text-xs text-gray-500">High ASR confidence</div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-sm text-gray-600">Feature Coverage</div>
            <div className="text-lg font-semibold text-gray-900">85%</div>
            <div className="text-xs text-gray-500">Most biomarkers extracted</div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-sm text-gray-600">Task Completion</div>
            <div className="text-lg font-semibold text-gray-900">100%</div>
            <div className="text-xs text-gray-500">All tasks completed</div>
          </div>
        </div>
      </div>

      {/* Quality Assurance Notes */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="text-xs text-gray-500 space-y-1">
          <p>• Quality assessment based on signal-to-noise ratio, transcription confidence, and feature completeness</p>
          <p>• Automated quality control with human review for borderline cases</p>
          <p>• Quality scores calibrated against research dataset standards</p>
        </div>
      </div>
    </div>
  )
}
