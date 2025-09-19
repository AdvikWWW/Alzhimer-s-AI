import React, { useState, useEffect } from 'react'
import { Users, Activity, AlertTriangle, CheckCircle, Clock, Eye, Filter, Download } from 'lucide-react'

interface AdminPageProps {}

interface SessionSummary {
  id: string
  participant_id: string
  created_at: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  quality_score: number
  requires_review: boolean
  risk_level: 'low' | 'moderate' | 'high'
}

interface QualityReview {
  id: string
  session_id: string
  participant_id: string
  flagged_at: string
  reasons: string[]
  priority: 'low' | 'medium' | 'high'
  assigned_to?: string
  status: 'pending' | 'in_review' | 'completed'
}

export const AdminPage: React.FC<AdminPageProps> = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'sessions' | 'quality' | 'system'>('dashboard')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [qualityReviews, setQualityReviews] = useState<QualityReview[]>([])
  const [loading, setLoading] = useState(false)
  const [filters, setFilters] = useState({
    status: 'all',
    dateRange: '7d',
    riskLevel: 'all'
  })

  // Mock data - in production, this would be fetched from the API
  useEffect(() => {
    const mockSessions: SessionSummary[] = [
      {
        id: 'sess_001',
        participant_id: 'part_001',
        created_at: '2024-01-15T10:30:00Z',
        status: 'completed',
        quality_score: 0.85,
        requires_review: false,
        risk_level: 'low'
      },
      {
        id: 'sess_002',
        participant_id: 'part_002',
        created_at: '2024-01-15T14:20:00Z',
        status: 'completed',
        quality_score: 0.62,
        requires_review: true,
        risk_level: 'high'
      },
      {
        id: 'sess_003',
        participant_id: 'part_003',
        created_at: '2024-01-16T09:15:00Z',
        status: 'processing',
        quality_score: 0.0,
        requires_review: false,
        risk_level: 'low'
      }
    ]

    const mockReviews: QualityReview[] = [
      {
        id: 'rev_001',
        session_id: 'sess_002',
        participant_id: 'part_002',
        flagged_at: '2024-01-15T14:25:00Z',
        reasons: ['Low audio quality', 'High uncertainty in ML predictions'],
        priority: 'high',
        status: 'pending'
      },
      {
        id: 'rev_002',
        session_id: 'sess_004',
        participant_id: 'part_004',
        flagged_at: '2024-01-16T11:30:00Z',
        reasons: ['Incomplete task completion', 'Borderline biomarker values'],
        priority: 'medium',
        status: 'in_review',
        assigned_to: 'Dr. Smith'
      }
    ]

    setSessions(mockSessions)
    setQualityReviews(mockReviews)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-50'
      case 'processing': return 'text-blue-600 bg-blue-50'
      case 'pending': return 'text-yellow-600 bg-yellow-50'
      case 'failed': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-600 bg-green-50'
      case 'moderate': return 'text-yellow-600 bg-yellow-50'
      case 'high': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'low': return 'text-blue-600 bg-blue-50'
      case 'medium': return 'text-yellow-600 bg-yellow-50'
      case 'high': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const handleAssignReview = (reviewId: string, assignee: string) => {
    setQualityReviews(prev => 
      prev.map(review => 
        review.id === reviewId 
          ? { ...review, assigned_to: assignee, status: 'in_review' as const }
          : review
      )
    )
  }

  const handleCompleteReview = (reviewId: string) => {
    setQualityReviews(prev => 
      prev.map(review => 
        review.id === reviewId 
          ? { ...review, status: 'completed' as const }
          : review
      )
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Monitor system performance, manage quality reviews, and oversee research data
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="border-b border-gray-200 mb-8">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: Activity },
              { id: 'sessions', label: 'Sessions', icon: Users },
              { id: 'quality', label: 'Quality Review', icon: Eye },
              { id: 'system', label: 'System Status', icon: CheckCircle }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === id
                    ? 'border-medical-500 text-medical-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="card">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <Users className="h-8 w-8 text-medical-600" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Total Sessions
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        1,247
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <AlertTriangle className="h-8 w-8 text-yellow-600" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Pending Reviews
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {qualityReviews.filter(r => r.status === 'pending').length}
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <Activity className="h-8 w-8 text-green-600" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Avg Quality Score
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        87.3%
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <Clock className="h-8 w-8 text-blue-600" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Processing Time
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        4.2 min
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="card">
              <h3 className="text-lg font-semibold mb-4">Recent Activity</h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Session sess_001 completed analysis</p>
                    <p className="text-xs text-gray-500">2 minutes ago</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                  <AlertTriangle className="h-5 w-5 text-yellow-600" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Session sess_002 flagged for review</p>
                    <p className="text-xs text-gray-500">15 minutes ago</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                  <Users className="h-5 w-5 text-blue-600" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">New participant registered</p>
                    <p className="text-xs text-gray-500">1 hour ago</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Sessions Tab */}
        {activeTab === 'sessions' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="card">
              <div className="flex items-center space-x-4">
                <Filter className="h-5 w-5 text-gray-400" />
                <select
                  value={filters.status}
                  onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
                  className="border border-gray-300 rounded px-3 py-1 text-sm"
                >
                  <option value="all">All Statuses</option>
                  <option value="completed">Completed</option>
                  <option value="processing">Processing</option>
                  <option value="pending">Pending</option>
                  <option value="failed">Failed</option>
                </select>
                
                <select
                  value={filters.riskLevel}
                  onChange={(e) => setFilters(prev => ({ ...prev, riskLevel: e.target.value }))}
                  className="border border-gray-300 rounded px-3 py-1 text-sm"
                >
                  <option value="all">All Risk Levels</option>
                  <option value="low">Low Risk</option>
                  <option value="moderate">Moderate Risk</option>
                  <option value="high">High Risk</option>
                </select>

                <button className="flex items-center space-x-2 px-3 py-1 bg-medical-600 text-white rounded text-sm hover:bg-medical-700">
                  <Download className="h-4 w-4" />
                  <span>Export</span>
                </button>
              </div>
            </div>

            {/* Sessions Table */}
            <div className="card">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Session ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Participant
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Created
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Quality Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Risk Level
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {sessions.map((session) => (
                      <tr key={session.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {session.id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {session.participant_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatDate(session.created_at)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(session.status)}`}>
                            {session.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {session.status === 'completed' ? `${Math.round(session.quality_score * 100)}%` : '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getRiskColor(session.risk_level)}`}>
                            {session.risk_level}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button className="text-medical-600 hover:text-medical-900 mr-3">
                            View
                          </button>
                          {session.requires_review && (
                            <button className="text-yellow-600 hover:text-yellow-900">
                              Review
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Quality Review Tab */}
        {activeTab === 'quality' && (
          <div className="space-y-6">
            <div className="card">
              <h3 className="text-lg font-semibold mb-4">Quality Review Queue</h3>
              <div className="space-y-4">
                {qualityReviews.map((review) => (
                  <div key={review.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <span className="font-medium text-gray-900">
                            Session {review.session_id}
                          </span>
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getPriorityColor(review.priority)}`}>
                            {review.priority} priority
                          </span>
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(review.status)}`}>
                            {review.status}
                          </span>
                        </div>
                        
                        <div className="text-sm text-gray-600 mb-2">
                          Participant: {review.participant_id} • Flagged: {formatDate(review.flagged_at)}
                        </div>
                        
                        <div className="mb-3">
                          <div className="text-sm font-medium text-gray-700 mb-1">Review Reasons:</div>
                          <ul className="text-sm text-gray-600 space-y-1">
                            {review.reasons.map((reason, index) => (
                              <li key={index} className="flex items-start space-x-1">
                                <span className="text-gray-400 mt-1">•</span>
                                <span>{reason}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        {review.assigned_to && (
                          <div className="text-sm text-gray-600">
                            Assigned to: <span className="font-medium">{review.assigned_to}</span>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex space-x-2 ml-4">
                        {review.status === 'pending' && (
                          <button
                            onClick={() => handleAssignReview(review.id, 'Current User')}
                            className="px-3 py-1 bg-medical-600 text-white text-sm rounded hover:bg-medical-700"
                          >
                            Assign to Me
                          </button>
                        )}
                        
                        {review.status === 'in_review' && (
                          <button
                            onClick={() => handleCompleteReview(review.id)}
                            className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700"
                          >
                            Complete Review
                          </button>
                        )}
                        
                        <button className="px-3 py-1 border border-gray-300 text-gray-700 text-sm rounded hover:bg-gray-50">
                          View Session
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* System Status Tab */}
        {activeTab === 'system' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* System Health */}
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">System Health</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">API Server</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="text-sm text-green-600">Healthy</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Database</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="text-sm text-green-600">Connected</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">ML Models</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                      <span className="text-sm text-yellow-600">Loading</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Storage</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="text-sm text-green-600">Available</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Performance</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">CPU Usage</span>
                    <span className="text-sm font-medium">23%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Memory Usage</span>
                    <span className="text-sm font-medium">67%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Disk Usage</span>
                    <span className="text-sm font-medium">45%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Active Sessions</span>
                    <span className="text-sm font-medium">12</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Model Status */}
            <div className="card">
              <h3 className="text-lg font-semibold mb-4">ML Model Status</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Model
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Version
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Last Updated
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Acoustic Classifier
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        v2.1.3
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-50 text-green-600">
                          Active
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        2024-01-10
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Lexical Analyzer
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        v1.8.2
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-50 text-green-600">
                          Active
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        2024-01-08
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        Ensemble Model
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        v3.0.1
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-yellow-50 text-yellow-600">
                          Loading
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        2024-01-15
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
