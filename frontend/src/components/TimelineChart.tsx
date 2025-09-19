import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Brush } from 'recharts'
import { Play, Pause, SkipBack, SkipForward, Volume2 } from 'lucide-react'

interface TimelineEvent {
  timestamp: number
  type: 'filled_pause' | 'silent_pause' | 'repetition' | 'false_start' | 'stutter'
  duration: number
  confidence: number
  text?: string
  severity: 'mild' | 'moderate' | 'severe'
}

interface TimelineChartProps {
  sessionId: string
  audioUrl?: string
}

export const TimelineChart: React.FC<TimelineChartProps> = ({ sessionId, audioUrl }) => {
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null)
  const [filterType, setFilterType] = useState<string>('all')

  // Mock timeline data - in production, this would be fetched from the API
  const timelineData = [
    { time: 2.5, filled_pause: 1, silent_pause: 0, repetition: 0, false_start: 0, stutter: 0 },
    { time: 8.2, filled_pause: 0, silent_pause: 1, repetition: 0, false_start: 0, stutter: 0 },
    { time: 15.7, filled_pause: 1, silent_pause: 0, repetition: 0, false_start: 0, stutter: 0 },
    { time: 23.1, filled_pause: 0, silent_pause: 0, repetition: 1, false_start: 0, stutter: 0 },
    { time: 31.4, filled_pause: 0, silent_pause: 1, repetition: 0, false_start: 0, stutter: 0 },
    { time: 38.9, filled_pause: 1, silent_pause: 0, repetition: 0, false_start: 0, stutter: 0 },
    { time: 45.2, filled_pause: 0, silent_pause: 0, repetition: 0, false_start: 1, stutter: 0 },
    { time: 52.8, filled_pause: 0, silent_pause: 1, repetition: 0, false_start: 0, stutter: 0 },
    { time: 61.3, filled_pause: 1, silent_pause: 0, repetition: 0, false_start: 0, stutter: 0 },
    { time: 68.7, filled_pause: 0, silent_pause: 0, repetition: 0, false_start: 0, stutter: 1 }
  ]

  const events: TimelineEvent[] = [
    { timestamp: 2.5, type: 'filled_pause', duration: 0.8, confidence: 0.92, text: 'uh', severity: 'mild' },
    { timestamp: 8.2, type: 'silent_pause', duration: 1.2, confidence: 0.88, severity: 'moderate' },
    { timestamp: 15.7, type: 'filled_pause', duration: 0.6, confidence: 0.95, text: 'um', severity: 'mild' },
    { timestamp: 23.1, type: 'repetition', duration: 1.4, confidence: 0.87, text: 'the the', severity: 'mild' },
    { timestamp: 31.4, type: 'silent_pause', duration: 2.1, confidence: 0.91, severity: 'severe' },
    { timestamp: 38.9, type: 'filled_pause', duration: 1.0, confidence: 0.89, text: 'uh', severity: 'moderate' },
    { timestamp: 45.2, type: 'false_start', duration: 1.8, confidence: 0.84, text: 'I was- I went', severity: 'moderate' },
    { timestamp: 52.8, type: 'silent_pause', duration: 1.5, confidence: 0.93, severity: 'moderate' },
    { timestamp: 61.3, type: 'filled_pause', duration: 0.9, confidence: 0.96, text: 'um', severity: 'mild' },
    { timestamp: 68.7, type: 'stutter', duration: 1.1, confidence: 0.82, text: 'b-b-but', severity: 'moderate' }
  ]

  const eventColors = {
    filled_pause: '#F59E0B',
    silent_pause: '#6B7280',
    repetition: '#EF4444',
    false_start: '#8B5CF6',
    stutter: '#EC4899'
  }

  const eventLabels = {
    filled_pause: 'Filled Pause',
    silent_pause: 'Silent Pause',
    repetition: 'Repetition',
    false_start: 'False Start',
    stutter: 'Stutter'
  }

  const filteredEvents = filterType === 'all' 
    ? events 
    : events.filter(event => event.type === filterType)

  const handleEventClick = (event: TimelineEvent) => {
    setSelectedEvent(event)
    setCurrentTime(event.timestamp)
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'mild': return 'bg-yellow-100 text-yellow-800'
      case 'moderate': return 'bg-orange-100 text-orange-800'
      case 'severe': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold">Disfluency Timeline</h3>
        <div className="flex space-x-2">
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded text-sm"
          >
            <option value="all">All Events</option>
            <option value="filled_pause">Filled Pauses</option>
            <option value="silent_pause">Silent Pauses</option>
            <option value="repetition">Repetitions</option>
            <option value="false_start">False Starts</option>
            <option value="stutter">Stutters</option>
          </select>
        </div>
      </div>

      {/* Audio Controls */}
      {audioUrl && (
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex items-center justify-center w-10 h-10 bg-medical-600 text-white rounded-full hover:bg-medical-700"
            >
              {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
            </button>
            
            <button className="p-2 text-gray-600 hover:text-gray-800">
              <SkipBack className="h-4 w-4" />
            </button>
            
            <button className="p-2 text-gray-600 hover:text-gray-800">
              <SkipForward className="h-4 w-4" />
            </button>
            
            <div className="flex-1">
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(75)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-medical-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(currentTime / 75) * 100}%` }}
                />
              </div>
            </div>
            
            <Volume2 className="h-4 w-4 text-gray-600" />
          </div>
        </div>
      )}

      {/* Timeline Chart */}
      <div className="mb-6">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="time" 
              type="number"
              scale="linear"
              domain={['dataMin', 'dataMax']}
              tickFormatter={formatTime}
            />
            <YAxis 
              domain={[0, 1]}
              tickFormatter={(value) => value === 1 ? 'Event' : ''}
            />
            <Tooltip 
              labelFormatter={(value) => `Time: ${formatTime(value as number)}`}
              formatter={(value: any, name: string) => {
                if (value === 1) {
                  return [eventLabels[name as keyof typeof eventLabels], 'Event Type']
                }
                return null
              }}
            />
            <ReferenceLine x={currentTime} stroke="#3B82F6" strokeWidth={2} />
            
            {Object.entries(eventColors).map(([type, color]) => (
              <Line
                key={type}
                type="stepAfter"
                dataKey={type}
                stroke={color}
                strokeWidth={3}
                dot={{ fill: color, strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, fill: color }}
              />
            ))}
            
            <Brush 
              dataKey="time" 
              height={30} 
              stroke="#3B82F6"
              tickFormatter={formatTime}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Event Legend */}
      <div className="flex flex-wrap gap-4 mb-6">
        {Object.entries(eventColors).map(([type, color]) => (
          <div key={type} className="flex items-center space-x-2">
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="text-sm text-gray-700">
              {eventLabels[type as keyof typeof eventLabels]}
            </span>
          </div>
        ))}
      </div>

      {/* Event List */}
      <div className="space-y-2">
        <h4 className="font-medium text-gray-900">Detected Events</h4>
        <div className="max-h-60 overflow-y-auto space-y-2">
          {filteredEvents.map((event, index) => (
            <div
              key={index}
              onClick={() => handleEventClick(event)}
              className={`p-3 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors ${
                selectedEvent === event ? 'border-medical-500 bg-medical-50' : 'border-gray-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: eventColors[event.type] }}
                  />
                  <div>
                    <div className="font-medium text-sm">
                      {eventLabels[event.type]}
                    </div>
                    <div className="text-xs text-gray-500">
                      {formatTime(event.timestamp)} • {event.duration.toFixed(1)}s
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 text-xs rounded-full ${getSeverityColor(event.severity)}`}>
                    {event.severity}
                  </span>
                  <span className="text-xs text-gray-500">
                    {Math.round(event.confidence * 100)}%
                  </span>
                </div>
              </div>
              
              {event.text && (
                <div className="mt-2 text-sm text-gray-700 italic">
                  "{event.text}"
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {Object.entries(eventLabels).map(([type, label]) => {
            const count = events.filter(e => e.type === type).length
            return (
              <div key={type} className="text-center">
                <div className="text-lg font-semibold text-gray-900">{count}</div>
                <div className="text-xs text-gray-500">{label}s</div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
