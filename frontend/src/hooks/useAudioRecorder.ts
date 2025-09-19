import { useState, useRef, useCallback, useEffect } from 'react'

export interface AudioRecorderState {
  isRecording: boolean
  isPaused: boolean
  recordingTime: number
  audioLevel: number
  error: string | null
}

export interface AudioRecorderControls {
  startRecording: () => Promise<void>
  stopRecording: () => Promise<Blob | null>
  pauseRecording: () => void
  resumeRecording: () => void
  resetRecording: () => void
}

export const useAudioRecorder = (): [AudioRecorderState, AudioRecorderControls] => {
  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    isPaused: false,
    recordingTime: 0,
    audioLevel: 0,
    error: null,
  })

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const intervalRef = useRef<number | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  // Audio level monitoring
  const monitorAudioLevel = useCallback(() => {
    if (!analyserRef.current) return

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)
    
    // Calculate RMS (Root Mean Square) for audio level
    let sum = 0
    for (let i = 0; i < dataArray.length; i++) {
      sum += dataArray[i] * dataArray[i]
    }
    const rms = Math.sqrt(sum / dataArray.length)
    const audioLevel = Math.min(rms / 128, 1) // Normalize to 0-1

    setState(prev => ({ ...prev, audioLevel }))
    
    if (state.isRecording && !state.isPaused) {
      animationFrameRef.current = requestAnimationFrame(monitorAudioLevel)
    }
  }, [state.isRecording, state.isPaused])

  // Timer for recording duration
  const startTimer = useCallback(() => {
    intervalRef.current = setInterval(() => {
      setState(prev => ({
        ...prev,
        recordingTime: prev.recordingTime + 1,
      }))
    }, 1000)
  }, [])

  const stopTimer = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  // Initialize audio context and analyser
  const setupAudioAnalysis = useCallback(async (stream: MediaStream) => {
    try {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyserRef.current = audioContextRef.current.createAnalyser()
      
      analyserRef.current.fftSize = 256
      analyserRef.current.smoothingTimeConstant = 0.8
      
      source.connect(analyserRef.current)
      
      // Start monitoring audio level
      monitorAudioLevel()
    } catch (error) {
      console.error('Error setting up audio analysis:', error)
    }
  }, [monitorAudioLevel])

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, error: null }))

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
        },
      })

      streamRef.current = stream
      audioChunksRef.current = []

      // Setup audio analysis
      await setupAudioAnalysis(stream)

      // Create MediaRecorder
      const options: MediaRecorderOptions = {}
      
      // Try to use high-quality audio formats
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        options.mimeType = 'audio/webm;codecs=opus'
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        options.mimeType = 'audio/mp4'
      } else if (MediaRecorder.isTypeSupported('audio/wav')) {
        options.mimeType = 'audio/wav'
      }

      mediaRecorderRef.current = new MediaRecorder(stream, options)

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorderRef.current.onerror = (event) => {
        setState(prev => ({ 
          ...prev, 
          error: 'Recording error occurred',
          isRecording: false 
        }))
      }

      // Start recording
      mediaRecorderRef.current.start(100) // Collect data every 100ms
      
      setState(prev => ({
        ...prev,
        isRecording: true,
        isPaused: false,
        recordingTime: 0,
      }))

      startTimer()

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start recording'
      setState(prev => ({ ...prev, error: errorMessage }))
      
      if (error instanceof Error && error.name === 'NotAllowedError') {
        setState(prev => ({ 
          ...prev, 
          error: 'Microphone access denied. Please allow microphone access and try again.' 
        }))
      }
    }
  }, [setupAudioAnalysis, startTimer])

  // Stop recording
  const stopRecording = useCallback(async (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current || !state.isRecording) {
        resolve(null)
        return
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: mediaRecorderRef.current?.mimeType || 'audio/wav' 
        })
        resolve(audioBlob)
      }

      mediaRecorderRef.current.stop()
      
      // Cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }

      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }

      stopTimer()

      setState(prev => ({
        ...prev,
        isRecording: false,
        isPaused: false,
        audioLevel: 0,
      }))
    })
  }, [state.isRecording, stopTimer])

  // Pause recording
  const pauseRecording = useCallback(() => {
    if (mediaRecorderRef.current && state.isRecording && !state.isPaused) {
      mediaRecorderRef.current.pause()
      stopTimer()
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }

      setState(prev => ({ ...prev, isPaused: true, audioLevel: 0 }))
    }
  }, [state.isRecording, state.isPaused, stopTimer])

  // Resume recording
  const resumeRecording = useCallback(() => {
    if (mediaRecorderRef.current && state.isRecording && state.isPaused) {
      mediaRecorderRef.current.resume()
      startTimer()
      monitorAudioLevel()
      setState(prev => ({ ...prev, isPaused: false }))
    }
  }, [state.isRecording, state.isPaused, startTimer, monitorAudioLevel])

  // Reset recording
  const resetRecording = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close()
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }

    stopTimer()
    
    audioChunksRef.current = []
    mediaRecorderRef.current = null
    streamRef.current = null
    audioContextRef.current = null
    analyserRef.current = null

    setState({
      isRecording: false,
      isPaused: false,
      recordingTime: 0,
      audioLevel: 0,
      error: null,
    })
  }, [stopTimer])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      resetRecording()
    }
  }, [resetRecording])

  return [
    state,
    {
      startRecording,
      stopRecording,
      pauseRecording,
      resumeRecording,
      resetRecording,
    },
  ]
}
