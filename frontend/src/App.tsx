import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { RecordingSession } from './pages/RecordingSession'
import { ResultsPage } from './pages/ResultsPage'
import { AdminPage } from './pages/AdminPage'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/session/:sessionId" element={<RecordingSession />} />
            <Route path="/results/:sessionId" element={<ResultsPage />} />
            <Route path="/admin" element={<AdminPage />} />
          </Routes>
        </Layout>
      </Router>
    </QueryClientProvider>
  )
}

export default App
