import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'
import { Activity, TrendingUp, Zap } from 'lucide-react'

interface BiomarkerChartProps {
  sessionId: string
}

export const BiomarkerChart: React.FC<BiomarkerChartProps> = ({ sessionId }) => {
  const [chartType, setChartType] = useState<'acoustic' | 'lexical' | 'disfluency'>('acoustic')
  const [loading, setLoading] = useState(false)

  // Mock biomarker data - in production, this would be fetched from the API
  const acousticData = [
    { feature: 'Pitch Mean', value: 185.4, normal_min: 150, normal_max: 220, unit: 'Hz' },
    { feature: 'Jitter', value: 1.2, normal_min: 0.5, normal_max: 1.5, unit: '%' },
    { feature: 'Shimmer', value: 4.5, normal_min: 2.0, normal_max: 5.0, unit: '%' },
    { feature: 'HNR', value: 18.6, normal_min: 15.0, normal_max: 25.0, unit: 'dB' },
    { feature: 'F1', value: 650, normal_min: 500, normal_max: 800, unit: 'Hz' },
    { feature: 'F2', value: 1720, normal_min: 1200, normal_max: 2000, unit: 'Hz' }
  ]

  const lexicalData = [
    { feature: 'TTR', value: 0.68, normal_min: 0.60, normal_max: 0.80, unit: '' },
    { feature: 'Semantic Coherence', value: 0.74, normal_min: 0.70, normal_max: 0.90, unit: '' },
    { feature: 'Idea Density', value: 0.58, normal_min: 0.50, normal_max: 0.70, unit: '' },
    { feature: 'Sentence Length', value: 8.4, normal_min: 7.0, normal_max: 12.0, unit: 'words' },
    { feature: 'Syntactic Complexity', value: 0.45, normal_min: 0.40, normal_max: 0.70, unit: '' }
  ]

  const disfluencyData = [
    { feature: 'Filled Pauses', value: 4.8, normal_min: 1.0, normal_max: 3.0, unit: '/100 words' },
    { feature: 'Silent Pauses', value: 8.3, normal_min: 4.0, normal_max: 7.0, unit: '/minute' },
    { feature: 'Repetitions', value: 1.2, normal_min: 0.5, normal_max: 2.0, unit: '/100 words' },
    { feature: 'Speech Rate', value: 4.2, normal_min: 4.5, normal_max: 6.0, unit: 'syll/sec' },
    { feature: 'Pause Duration', value: 0.85, normal_min: 0.50, normal_max: 0.80, unit: 'seconds' }
  ]

  const getCurrentData = () => {
    switch (chartType) {
      case 'acoustic': return acousticData
      case 'lexical': return lexicalData
      case 'disfluency': return disfluencyData
      default: return acousticData
    }
  }

  const getStatusColor = (value: number, min: number, max: number) => {
    if (value >= min && value <= max) return '#10B981' // Green - normal
    if (value < min * 0.8 || value > max * 1.2) return '#EF4444' // Red - significantly abnormal
    return '#F59E0B' // Yellow - borderline
  }

  const getStatusText = (value: number, min: number, max: number) => {
    if (value >= min && value <= max) return 'Normal'
    if (value < min * 0.8 || value > max * 1.2) return 'Abnormal'
    return 'Borderline'
  }

  // Radar chart data for comprehensive view
  const radarData = [
    { subject: 'Acoustic', A: 75, B: 80, fullMark: 100 },
    { subject: 'Lexical', A: 68, B: 80, fullMark: 100 },
    { subject: 'Semantic', A: 74, B: 80, fullMark: 100 },
    { subject: 'Fluency', A: 62, B: 80, fullMark: 100 },
    { subject: 'Voice Quality', A: 71, B: 80, fullMark: 100 },
    { subject: 'Timing', A: 58, B: 80, fullMark: 100 }
  ]

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold">Biomarker Analysis</h3>
        <div className="flex space-x-2">
          <button
            onClick={() => setChartType('acoustic')}
            className={`px-3 py-1 rounded text-sm font-medium ${
              chartType === 'acoustic' 
                ? 'bg-medical-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Activity className="h-4 w-4 inline mr-1" />
            Acoustic
          </button>
          <button
            onClick={() => setChartType('lexical')}
            className={`px-3 py-1 rounded text-sm font-medium ${
              chartType === 'lexical' 
                ? 'bg-medical-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <TrendingUp className="h-4 w-4 inline mr-1" />
            Lexical
          </button>
          <button
            onClick={() => setChartType('disfluency')}
            className={`px-3 py-1 rounded text-sm font-medium ${
              chartType === 'disfluency' 
                ? 'bg-medical-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Zap className="h-4 w-4 inline mr-1" />
            Disfluency
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Bar Chart */}
        <div>
          <h4 className="font-medium text-gray-900 mb-4">
            {chartType.charAt(0).toUpperCase() + chartType.slice(1)} Features
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={getCurrentData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="feature" 
                angle={-45}
                textAnchor="end"
                height={80}
                fontSize={12}
              />
              <YAxis fontSize={12} />
              <Tooltip 
                formatter={(value: any, name: string, props: any) => [
                  `${value} ${props.payload.unit}`,
                  'Value'
                ]}
                labelFormatter={(label) => `Feature: ${label}`}
              />
              <Bar 
                dataKey="value" 
                fill="#3B82F6"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart - Overall Profile */}
        <div>
          <h4 className="font-medium text-gray-900 mb-4">Biomarker Profile</h4>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" fontSize={12} />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 100]} 
                fontSize={10}
                tickCount={5}
              />
              <Radar
                name="Patient"
                dataKey="A"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Radar
                name="Normal Range"
                dataKey="B"
                stroke="#10B981"
                fill="transparent"
                strokeWidth={2}
                strokeDasharray="5 5"
              />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Feature Table */}
      <div className="mt-6">
        <h4 className="font-medium text-gray-900 mb-4">Feature Details</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Feature
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Normal Range
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {getCurrentData().map((item, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {item.feature}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {item.value} {item.unit}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item.normal_min} - {item.normal_max} {item.unit}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span 
                      className="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
                      style={{ 
                        backgroundColor: `${getStatusColor(item.value, item.normal_min, item.normal_max)}20`,
                        color: getStatusColor(item.value, item.normal_min, item.normal_max)
                      }}
                    >
                      {getStatusText(item.value, item.normal_min, item.normal_max)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Clinical Interpretation */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="font-medium text-blue-900 mb-2">Clinical Interpretation</h4>
        <div className="text-sm text-blue-800 space-y-1">
          {chartType === 'acoustic' && (
            <>
              <p>• Pitch parameters within normal range for age and gender</p>
              <p>• Slight elevation in jitter suggesting mild voice instability</p>
              <p>• Formant frequencies indicate normal vocal tract configuration</p>
            </>
          )}
          {chartType === 'lexical' && (
            <>
              <p>• Good lexical diversity with preserved vocabulary access</p>
              <p>• Maintained semantic coherence throughout tasks</p>
              <p>• Moderate syntactic complexity appropriate for cognitive status</p>
            </>
          )}
          {chartType === 'disfluency' && (
            <>
              <p>• Elevated filled pause rate may indicate word-finding difficulty</p>
              <p>• Increased silent pausing frequency suggests processing delays</p>
              <p>• Reduced speech rate consistent with cognitive changes</p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
