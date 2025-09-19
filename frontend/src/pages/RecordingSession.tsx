import React, { useState, useEffect } from 'react'
import { useParams, useLocation, useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, CheckCircle, Clock, FileAudio } from 'lucide-react'
import { AudioRecorder } from '../components/AudioRecorder'

interface RecordingTask {
  id: string
  title: string
  description: string
  instructions: string[]
  maxDuration: number
  type: 'narrative' | 'picture' | 'fluency' | 'reading' | 'repetition' | 'naming' | 'conversation' | 'calibration'
  stimulus?: string
  imageUrl?: string
}

const RECORDING_TASKS: RecordingTask[] = [
  {
    id: 'calibration',
    title: 'Microphone Calibration',
    description: 'Test your microphone and adjust volume levels',
    instructions: [
      'Count from 1 to 10 at your normal speaking volume',
      'This helps us calibrate the audio levels for optimal recording quality',
      'Speak clearly and maintain consistent distance from the microphone'
    ],
    maxDuration: 30,
    type: 'calibration'
  },
  {
    id: 'narrative',
    title: 'Free Narrative',
    description: 'Describe a recent memorable event in your life',
    instructions: [
      'Think of a recent event that was meaningful or memorable to you',
      'Describe what happened, when it occurred, and why it was significant',
      'Speak naturally and take your time - aim for 2-3 minutes',
      'Include details about the people involved and your feelings about the event'
    ],
    maxDuration: 240,
    type: 'narrative'
  },
  {
    id: 'picture',
    title: 'Picture Description',
    description: 'Describe what you see in the Cookie Theft picture',
    instructions: [
      'Look at the picture carefully',
      'Describe everything you can see happening in the scene',
      'Include details about the people, objects, and actions',
      'Speak for about 2 minutes, describing as much as you can'
    ],
    maxDuration: 180,
    type: 'picture',
    imageUrl: '/images/cookie-theft.jpg'
  },
  {
    id: 'fluency',
    title: 'Semantic Fluency',
    description: 'Name as many animals as you can in 60 seconds',
    instructions: [
      'When the recording starts, name as many different animals as you can',
      'You have exactly 60 seconds',
      'Say each animal clearly and try not to repeat any',
      'Any type of animal counts - pets, wild animals, sea creatures, etc.'
    ],
    maxDuration: 60,
    type: 'fluency'
  },
  {
    id: 'reading',
    title: 'Reading Passage',
    description: 'Read the Rainbow Passage aloud',
    instructions: [
      'Read the passage below at your normal reading pace',
      'Speak clearly and naturally',
      'Don\'t worry about making small mistakes - just continue reading',
      'Take your time to pronounce each word clearly'
    ],
    maxDuration: 120,
    type: 'reading',
    stimulus: `When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.`
  },
  {
    id: 'repetition',
    title: 'Sentence Repetition',
    description: 'Listen and repeat each sentence exactly as you hear it',
    instructions: [
      'You will hear 6 sentences of increasing complexity',
      'After each sentence, repeat it back exactly as you heard it',
      'Try to match the rhythm and intonation as well as the words',
      'Take your time and speak clearly'
    ],
    maxDuration: 180,
    type: 'repetition'
  },
  {
    id: 'naming',
    title: 'Confrontation Naming',
    description: 'Name the objects you see in each image',
    instructions: [
      'You will see 15 common objects one at a time',
      'Name each object as quickly and accurately as possible',
      'Say just the name of the object - no need for descriptions',
      'If you\'re unsure, give your best guess'
    ],
    maxDuration: 120,
    type: 'naming'
  },
  {
    id: 'conversation1',
    title: 'Conversational Prompt 1',
    description: 'Tell me about your typical day',
    instructions: [
      'Describe what a typical day looks like for you',
      'Include your daily routines, activities, and interactions',
      'Speak naturally as if you\'re talking to a friend',
      'Aim for about 45 seconds of speaking'
    ],
    maxDuration: 60,
    type: 'conversation'
  },
  {
    id: 'conversation2',
    title: 'Conversational Prompt 2',
    description: 'What are your thoughts on modern technology?',
    instructions: [
      'Share your opinions about how technology has changed our lives',
      'You might discuss smartphones, computers, social media, etc.',
      'Express both positive and negative aspects if you have them',
      'Speak naturally for about 45 seconds'
    ],
    maxDuration: 60,
    type: 'conversation'
  }
]

export const RecordingSession: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>()
  const location = useLocation()
  const navigate = useNavigate()
  
  const [currentTaskIndex, setCurrentTaskIndex] = useState(0)
  const [completedTasks, setCompletedTasks] = useState<Set<string>>(new Set())
  const [recordings, setRecordings] = useState<Map<string, Blob>>(new Map())
  const [sessionStartTime] = useState(new Date())

  const participantData = location.state?.participantData
  const currentTask = RECORDING_TASKS[currentTaskIndex]
  const isLastTask = currentTaskIndex === RECORDING_TASKS.length - 1
  const canProceed = completedTasks.has(currentTask.id)

  useEffect(() => {
    if (!participantData) {
      navigate('/')
    }
  }, [participantData, navigate])

  const handleRecordingComplete = (audioBlob: Blob) => {
    setRecordings(prev => new Map(prev).set(currentTask.id, audioBlob))
    setCompletedTasks(prev => new Set(prev).add(currentTask.id))
  }

  const handleNextTask = () => {
    if (currentTaskIndex < RECORDING_TASKS.length - 1) {
      setCurrentTaskIndex(prev => prev + 1)
    }
  }

  const handlePreviousTask = () => {
    if (currentTaskIndex > 0) {
      setCurrentTaskIndex(prev => prev - 1)
    }
  }

  const handleCompleteSession = async () => {
    // Upload recordings and participant data
    const sessionData = {
      sessionId,
      participantData,
      recordings: Array.from(recordings.entries()),
      completedTasks: Array.from(completedTasks),
      sessionStartTime,
      sessionEndTime: new Date(),
    }

    // TODO: Implement API call to upload session data
    console.log('Session completed:', sessionData)
    
    navigate(`/results/${sessionId}`, { state: { sessionData } })
  }

  const getTaskIcon = (type: string) => {
    switch (type) {
      case 'calibration': return '🎙️'
      case 'narrative': return '📖'
      case 'picture': return '🖼️'
      case 'fluency': return '🐾'
      case 'reading': return '📚'
      case 'repetition': return '🔄'
      case 'naming': return '🏷️'
      case 'conversation': return '💬'
      default: return '📝'
    }
  }

  if (!participantData) {
    return <div>Loading...</div>
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Session Header */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Recording Session</h1>
            <p className="text-gray-600">Participant: {participantData.participantId}</p>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Session ID</div>
            <div className="font-mono text-sm">{sessionId}</div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Progress</span>
            <span>{completedTasks.size} of {RECORDING_TASKS.length} tasks completed</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-medical-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(completedTasks.size / RECORDING_TASKS.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Task Navigation */}
        <div className="flex items-center justify-between">
          <button
            onClick={handlePreviousTask}
            disabled={currentTaskIndex === 0}
            className={`btn-secondary flex items-center space-x-2 ${
              currentTaskIndex === 0 ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            <ChevronLeft className="h-4 w-4" />
            <span>Previous</span>
          </button>

          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-500">Task</span>
            <span className="font-semibold">{currentTaskIndex + 1}</span>
            <span className="text-gray-400">of</span>
            <span className="font-semibold">{RECORDING_TASKS.length}</span>
          </div>

          {!isLastTask ? (
            <button
              onClick={handleNextTask}
              disabled={!canProceed}
              className={`btn-secondary flex items-center space-x-2 ${
                !canProceed ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              <span>Next</span>
              <ChevronRight className="h-4 w-4" />
            </button>
          ) : (
            <button
              onClick={handleCompleteSession}
              disabled={completedTasks.size !== RECORDING_TASKS.length}
              className={`btn-medical flex items-center space-x-2 ${
                completedTasks.size !== RECORDING_TASKS.length ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              <CheckCircle className="h-4 w-4" />
              <span>Complete Session</span>
            </button>
          )}
        </div>
      </div>

      {/* Current Task */}
      <div className="card">
        <div className="flex items-start space-x-4 mb-6">
          <div className="text-3xl">{getTaskIcon(currentTask.type)}</div>
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-2">
              <h2 className="text-xl font-bold text-gray-900">{currentTask.title}</h2>
              {completedTasks.has(currentTask.id) && (
                <CheckCircle className="h-5 w-5 text-green-600" />
              )}
            </div>
            <p className="text-gray-600 mb-4">{currentTask.description}</p>
            
            <div className="flex items-center space-x-4 text-sm text-gray-500 mb-4">
              <div className="flex items-center space-x-1">
                <Clock className="h-4 w-4" />
                <span>Max {Math.floor(currentTask.maxDuration / 60)}:{(currentTask.maxDuration % 60).toString().padStart(2, '0')}</span>
              </div>
              <div className="flex items-center space-x-1">
                <FileAudio className="h-4 w-4" />
                <span>High quality audio</span>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-medium text-blue-900 mb-2">Instructions:</h3>
              <ul className="space-y-1">
                {currentTask.instructions.map((instruction, index) => (
                  <li key={index} className="text-blue-800 text-sm">
                    {instruction}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Task-specific content */}
        {currentTask.imageUrl && (
          <div className="mb-6">
            <img
              src={currentTask.imageUrl}
              alt="Task stimulus"
              className="max-w-md mx-auto rounded-lg border border-gray-200"
            />
          </div>
        )}

        {currentTask.stimulus && (
          <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <h3 className="font-medium text-gray-900 mb-2">Reading Passage:</h3>
            <p className="text-gray-800 leading-relaxed">{currentTask.stimulus}</p>
          </div>
        )}

        {/* Audio Recorder */}
        <AudioRecorder
          onRecordingComplete={handleRecordingComplete}
          maxDuration={currentTask.maxDuration}
          className="mt-6"
        />
      </div>

      {/* Task List Sidebar */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">All Tasks</h3>
        <div className="space-y-2">
          {RECORDING_TASKS.map((task, index) => (
            <div
              key={task.id}
              className={`flex items-center space-x-3 p-3 rounded-lg cursor-pointer transition-colors ${
                index === currentTaskIndex
                  ? 'bg-medical-50 border border-medical-200'
                  : completedTasks.has(task.id)
                  ? 'bg-green-50 border border-green-200'
                  : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
              }`}
              onClick={() => setCurrentTaskIndex(index)}
            >
              <div className="text-lg">{getTaskIcon(task.type)}</div>
              <div className="flex-1">
                <div className="font-medium text-sm">{task.title}</div>
                <div className="text-xs text-gray-500">{Math.floor(task.maxDuration / 60)}min max</div>
              </div>
              {completedTasks.has(task.id) && (
                <CheckCircle className="h-4 w-4 text-green-600" />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
