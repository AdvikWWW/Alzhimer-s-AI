import React from 'react'
import { AlertTriangle, TrendingUp, Shield, Activity } from 'lucide-react'

interface RiskAssessmentProps {
  riskAssessment: {
    risk_probability: number
    confidence_interval_lower: number
    confidence_interval_upper: number
    uncertainty_score: number
    risk_category: string
  }
}

export const RiskAssessment: React.FC<RiskAssessmentProps> = ({ riskAssessment }) => {
  const getRiskColor = (category: string) => {
    switch (category) {
      case 'low': return 'text-green-600'
      case 'moderate': return 'text-yellow-600'
      case 'high': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getRiskIcon = (category: string) => {
    switch (category) {
      case 'low': return Shield
      case 'moderate': return Activity
      case 'high': return AlertTriangle
      default: return TrendingUp
    }
  }

  const RiskIcon = getRiskIcon(riskAssessment.risk_category)
  
  return (
    <div className="card">
      <div className="flex items-center space-x-3 mb-6">
        <RiskIcon className={`h-6 w-6 ${getRiskColor(riskAssessment.risk_category)}`} />
        <h3 className="text-lg font-semibold">Risk Assessment</h3>
      </div>

      {/* Risk Probability Gauge */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Risk Probability</span>
          <span className="text-lg font-bold">
            {Math.round(riskAssessment.risk_probability * 100)}%
          </span>
        </div>
        
        <div className="relative">
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all duration-300 ${
                riskAssessment.risk_category === 'low' ? 'bg-green-500' :
                riskAssessment.risk_category === 'moderate' ? 'bg-yellow-500' :
                'bg-red-500'
              }`}
              style={{ width: `${riskAssessment.risk_probability * 100}%` }}
            />
          </div>
          
          {/* Confidence Interval Markers */}
          <div className="absolute top-0 h-3 w-full">
            <div 
              className="absolute top-0 bottom-0 bg-black bg-opacity-20 rounded-full"
              style={{ 
                left: `${riskAssessment.confidence_interval_lower * 100}%`,
                width: `${(riskAssessment.confidence_interval_upper - riskAssessment.confidence_interval_lower) * 100}%`
              }}
            />
          </div>
        </div>
        
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>

      {/* Confidence Interval */}
      <div className="mb-6">
        <div className="text-sm font-medium text-gray-700 mb-2">95% Confidence Interval</div>
        <div className="flex items-center space-x-2">
          <div className="bg-gray-100 px-3 py-1 rounded text-sm">
            {Math.round(riskAssessment.confidence_interval_lower * 100)}%
          </div>
          <span className="text-gray-400">—</span>
          <div className="bg-gray-100 px-3 py-1 rounded text-sm">
            {Math.round(riskAssessment.confidence_interval_upper * 100)}%
          </div>
        </div>
      </div>

      {/* Uncertainty Score */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Model Uncertainty</span>
          <span className="text-sm font-medium">
            {Math.round(riskAssessment.uncertainty_score * 100)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className={`h-2 rounded-full ${
              riskAssessment.uncertainty_score < 0.3 ? 'bg-green-400' :
              riskAssessment.uncertainty_score < 0.6 ? 'bg-yellow-400' :
              'bg-red-400'
            }`}
            style={{ width: `${riskAssessment.uncertainty_score * 100}%` }}
          />
        </div>
      </div>

      {/* Risk Category */}
      <div className={`border rounded-lg p-4 ${
        riskAssessment.risk_category === 'low' ? 'bg-green-50 border-green-200' :
        riskAssessment.risk_category === 'moderate' ? 'bg-yellow-50 border-yellow-200' :
        'bg-red-50 border-red-200'
      }`}>
        <div className="flex items-center space-x-2 mb-2">
          <RiskIcon className={`h-4 w-4 ${getRiskColor(riskAssessment.risk_category)}`} />
          <span className={`font-medium ${getRiskColor(riskAssessment.risk_category)}`}>
            {riskAssessment.risk_category.charAt(0).toUpperCase() + riskAssessment.risk_category.slice(1)} Risk
          </span>
        </div>
        <p className="text-sm text-gray-700">
          {riskAssessment.risk_category === 'low' && 
            'Voice biomarkers suggest minimal risk of cognitive impairment. Continue routine monitoring.'}
          {riskAssessment.risk_category === 'moderate' && 
            'Voice biomarkers indicate possible early cognitive changes. Consider follow-up assessment.'}
          {riskAssessment.risk_category === 'high' && 
            'Voice biomarkers suggest significant risk of cognitive impairment. Comprehensive evaluation recommended.'}
        </p>
      </div>

      {/* Model Information */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="text-xs text-gray-500 space-y-1">
          <p>• Ensemble model combining acoustic, lexical, and semantic features</p>
          <p>• Trained on DementiaBank and ADReSS research datasets</p>
          <p>• Calibrated probabilities with uncertainty quantification</p>
        </div>
      </div>
    </div>
  )
}
