import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Brain, Home, Settings, FileText } from 'lucide-react'

interface LayoutProps {
  children: React.ReactNode
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation()

  const navigation = [
    { name: 'Home', href: '/', icon: Home },
    { name: 'Results', href: '/results', icon: FileText },
    { name: 'Admin', href: '/admin', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-medical-600" />
              <h1 className="ml-3 text-xl font-semibold text-gray-900">
                Voice Biomarker Platform
              </h1>
            </div>
            
            <nav className="flex space-x-8">
              {navigation.map((item) => {
                const Icon = item.icon
                const isActive = location.pathname === item.href
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'text-medical-600 bg-medical-50'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className="h-4 w-4 mr-2" />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-gray-500">
            <p className="font-medium text-red-600 mb-2">
              ⚠️ FOR RESEARCH USE ONLY
            </p>
            <p>
              This application is designed for research purposes only and should not be used for clinical diagnosis.
              All results include uncertainty measures and require professional interpretation.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
