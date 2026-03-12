import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar/Navbar'
import Dashboard from './pages/Dashboard'
import Training from './pages/Training'
import Backtesting from './pages/Backtesting'
import Predictions from './pages/Predictions'
import './App.css'

function App() {
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/health')
        if (response.ok) {
          setApiStatus('connected')
        } else {
          setApiStatus('error')
        }
      } catch (error) {
        setApiStatus('disconnected')
      }
    }

    checkApiHealth()
    const interval = setInterval(checkApiHealth, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <Router>
      <div className="app">
        <Navbar apiStatus={apiStatus} />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/training" element={<Training />} />
            <Route path="/backtesting" element={<Backtesting />} />
            <Route path="/predictions" element={<Predictions />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
