import React, { useState, useEffect } from 'react'
import { Mic, Square, Pause, Play, RotateCcw } from 'lucide-react'
import { useAudioRecorder } from '../hooks/useAudioRecorder'

interface AudioRecorderProps {
  onRecordingComplete: (audioBlob: Blob) => void
  onRecordingStart?: () => void
  maxDuration?: number // in seconds
  disabled?: boolean
  className?: string
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  onRecordingComplete,
  onRecordingStart,
  maxDuration = 300, // 5 minutes default
  disabled = false,
  className = '',
}) => {
  const [recordingState, recordingControls] = useAudioRecorder()
  const [hasRecorded, setHasRecorded] = useState(false)

  // Auto-stop recording when max duration is reached
  useEffect(() => {
    if (recordingState.recordingTime >= maxDuration && recordingState.isRecording) {
      handleStopRecording()
    }
  }, [recordingState.recordingTime, maxDuration, recordingState.isRecording])

  const handleStartRecording = async () => {
    try {
      await recordingControls.startRecording()
      setHasRecorded(false)
      onRecordingStart?.()
    } catch (error) {
      console.error('Failed to start recording:', error)
    }
  }

  const handleStopRecording = async () => {
    const audioBlob = await recordingControls.stopRecording()
    if (audioBlob) {
      setHasRecorded(true)
      onRecordingComplete(audioBlob)
    }
  }

  const handlePauseResume = () => {
    if (recordingState.isPaused) {
      recordingControls.resumeRecording()
    } else {
      recordingControls.pauseRecording()
    }
  }

  const handleReset = () => {
    recordingControls.resetRecording()
    setHasRecorded(false)
  }

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const getAudioLevelHeight = (): string => {
    return `${Math.max(recordingState.audioLevel * 100, 2)}%`
  }

  const remainingTime = maxDuration - recordingState.recordingTime

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-6 ${className}`}>
      {/* Audio Level Visualizer */}
      <div className="mb-6">
        <div className="flex items-center justify-center h-20 bg-gray-50 rounded-lg relative overflow-hidden">
          {recordingState.isRecording && !recordingState.isPaused ? (
            <div className="flex items-end justify-center space-x-1 h-full">
              {Array.from({ length: 20 }, (_, i) => (
                <div
                  key={i}
                  className="bg-medical-500 rounded-t transition-all duration-75"
                  style={{
                    width: '8px',
                    height: getAudioLevelHeight(),
                    opacity: Math.random() * 0.5 + 0.5, // Random opacity for visual effect
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="text-gray-400 text-sm">
              {hasRecorded ? 'Recording completed' : 'Audio visualization will appear here'}
            </div>
          )}
        </div>
      </div>

      {/* Recording Status */}
      <div className="text-center mb-6">
        <div className="flex items-center justify-center space-x-4 mb-2">
          {recordingState.isRecording && (
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${recordingState.isPaused ? 'bg-yellow-500' : 'bg-red-500 animate-pulse'}`} />
              <span className="text-sm font-medium">
                {recordingState.isPaused ? 'Paused' : 'Recording'}
              </span>
            </div>
          )}
        </div>
        
        <div className="text-2xl font-mono font-bold text-gray-900 mb-1">
          {formatTime(recordingState.recordingTime)}
        </div>
        
        {recordingState.isRecording && remainingTime > 0 && (
          <div className="text-sm text-gray-500">
            {formatTime(remainingTime)} remaining
          </div>
        )}
        
        {remainingTime <= 30 && recordingState.isRecording && (
          <div className="text-sm text-orange-600 font-medium">
            Recording will auto-stop in {remainingTime}s
          </div>
        )}
      </div>

      {/* Error Display */}
      {recordingState.error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700 text-sm">{recordingState.error}</p>
        </div>
      )}

      {/* Controls */}
      <div className="flex justify-center space-x-3">
        {!recordingState.isRecording ? (
          <>
            <button
              onClick={handleStartRecording}
              disabled={disabled}
              className={`btn-medical flex items-center space-x-2 ${
                disabled ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              <Mic className="h-5 w-5" />
              <span>Start Recording</span>
            </button>
            
            {hasRecorded && (
              <button
                onClick={handleReset}
                className="btn-secondary flex items-center space-x-2"
              >
                <RotateCcw className="h-4 w-4" />
                <span>Reset</span>
              </button>
            )}
          </>
        ) : (
          <>
            <button
              onClick={handlePauseResume}
              className="btn-secondary flex items-center space-x-2"
            >
              {recordingState.isPaused ? (
                <>
                  <Play className="h-4 w-4" />
                  <span>Resume</span>
                </>
              ) : (
                <>
                  <Pause className="h-4 w-4" />
                  <span>Pause</span>
                </>
              )}
            </button>
            
            <button
              onClick={handleStopRecording}
              className="btn-primary flex items-center space-x-2 bg-red-600 hover:bg-red-700"
            >
              <Square className="h-4 w-4" />
              <span>Stop</span>
            </button>
          </>
        )}
      </div>

      {/* Recording Tips */}
      {!recordingState.isRecording && !hasRecorded && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">Recording Tips:</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Ensure you're in a quiet environment</li>
            <li>• Speak clearly and at a normal pace</li>
            <li>• Keep the microphone at a consistent distance</li>
            <li>• You can pause and resume if needed</li>
          </ul>
        </div>
      )}
    </div>
  )
}
