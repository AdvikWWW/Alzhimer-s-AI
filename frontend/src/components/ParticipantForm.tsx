import React from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { User, Calendar, FileText } from 'lucide-react'

const participantSchema = z.object({
  participantId: z.string().min(1, 'Participant ID is required'),
  age: z.number().min(18, 'Age must be at least 18').max(120, 'Age must be realistic'),
  gender: z.enum(['male', 'female', 'other', 'prefer-not-to-say']),
  nativeLanguage: z.string().min(1, 'Native language is required'),
  educationLevel: z.enum(['less-than-high-school', 'high-school', 'some-college', 'bachelors', 'masters', 'doctorate']),
  hearingImpairment: z.boolean(),
  speechImpairment: z.boolean(),
  cognitiveStatus: z.enum(['healthy-control', 'mild-cognitive-impairment', 'alzheimers-disease', 'other-dementia', 'unknown']),
  medications: z.string().optional(),
  consentGiven: z.boolean().refine(val => val === true, 'Consent must be given to proceed'),
  notes: z.string().optional(),
})

type ParticipantData = z.infer<typeof participantSchema>

interface ParticipantFormProps {
  onSubmit: (data: ParticipantData) => void
}

export const ParticipantForm: React.FC<ParticipantFormProps> = ({ onSubmit }) => {
  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<ParticipantData>({
    resolver: zodResolver(participantSchema),
    mode: 'onChange',
  })

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card">
        <div className="flex items-center mb-6">
          <User className="h-6 w-6 text-medical-600 mr-3" />
          <h2 className="text-2xl font-bold">Participant Information</h2>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* Basic Information */}
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Participant ID *
              </label>
              <input
                {...register('participantId')}
                className="input-field"
                placeholder="e.g., P001"
              />
              {errors.participantId && (
                <p className="text-red-600 text-sm mt-1">{errors.participantId.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Age *
              </label>
              <input
                {...register('age', { valueAsNumber: true })}
                type="number"
                className="input-field"
                placeholder="e.g., 65"
              />
              {errors.age && (
                <p className="text-red-600 text-sm mt-1">{errors.age.message}</p>
              )}
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Gender *
              </label>
              <select {...register('gender')} className="input-field">
                <option value="">Select gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
                <option value="prefer-not-to-say">Prefer not to say</option>
              </select>
              {errors.gender && (
                <p className="text-red-600 text-sm mt-1">{errors.gender.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Native Language *
              </label>
              <input
                {...register('nativeLanguage')}
                className="input-field"
                placeholder="e.g., English"
              />
              {errors.nativeLanguage && (
                <p className="text-red-600 text-sm mt-1">{errors.nativeLanguage.message}</p>
              )}
            </div>
          </div>

          {/* Education and Health */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Education Level *
            </label>
            <select {...register('educationLevel')} className="input-field">
              <option value="">Select education level</option>
              <option value="less-than-high-school">Less than high school</option>
              <option value="high-school">High school diploma/GED</option>
              <option value="some-college">Some college</option>
              <option value="bachelors">Bachelor's degree</option>
              <option value="masters">Master's degree</option>
              <option value="doctorate">Doctorate degree</option>
            </select>
            {errors.educationLevel && (
              <p className="text-red-600 text-sm mt-1">{errors.educationLevel.message}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Cognitive Status *
            </label>
            <select {...register('cognitiveStatus')} className="input-field">
              <option value="">Select cognitive status</option>
              <option value="healthy-control">Healthy control</option>
              <option value="mild-cognitive-impairment">Mild cognitive impairment</option>
              <option value="alzheimers-disease">Alzheimer's disease</option>
              <option value="other-dementia">Other dementia</option>
              <option value="unknown">Unknown/Not assessed</option>
            </select>
            {errors.cognitiveStatus && (
              <p className="text-red-600 text-sm mt-1">{errors.cognitiveStatus.message}</p>
            )}
          </div>

          {/* Health Conditions */}
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900">Health Conditions</h3>
            
            <div className="flex items-center">
              <input
                {...register('hearingImpairment')}
                type="checkbox"
                className="h-4 w-4 text-medical-600 focus:ring-medical-500 border-gray-300 rounded"
              />
              <label className="ml-2 text-sm text-gray-700">
                Hearing impairment or use of hearing aids
              </label>
            </div>

            <div className="flex items-center">
              <input
                {...register('speechImpairment')}
                type="checkbox"
                className="h-4 w-4 text-medical-600 focus:ring-medical-500 border-gray-300 rounded"
              />
              <label className="ml-2 text-sm text-gray-700">
                Speech impairment or speech therapy history
              </label>
            </div>
          </div>

          {/* Optional Fields */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Current Medications (optional)
            </label>
            <textarea
              {...register('medications')}
              className="input-field"
              rows={3}
              placeholder="List any medications that might affect speech or cognition..."
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Additional Notes (optional)
            </label>
            <textarea
              {...register('notes')}
              className="input-field"
              rows={3}
              placeholder="Any additional relevant information..."
            />
          </div>

          {/* Consent */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start">
              <input
                {...register('consentGiven')}
                type="checkbox"
                className="h-4 w-4 text-medical-600 focus:ring-medical-500 border-gray-300 rounded mt-1"
              />
              <div className="ml-3">
                <label className="text-sm font-medium text-gray-900">
                  Research Consent *
                </label>
                <p className="text-sm text-gray-600 mt-1">
                  I understand that this voice recording session is for research purposes only. 
                  The data collected will be used to develop and validate voice biomarkers for 
                  Alzheimer's disease research. I consent to the recording and analysis of my voice 
                  for this research purpose.
                </p>
              </div>
            </div>
            {errors.consentGiven && (
              <p className="text-red-600 text-sm mt-2">{errors.consentGiven.message}</p>
            )}
          </div>

          {/* Submit Button */}
          <div className="flex justify-center pt-4">
            <button
              type="submit"
              disabled={!isValid}
              className={`btn-medical text-lg px-8 py-3 ${
                !isValid ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              <Calendar className="h-5 w-5 mr-2 inline" />
              Begin Assessment Session
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
