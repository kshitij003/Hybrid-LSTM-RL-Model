import React, { useState } from 'react'
import Card from '../components/Card/Card'
import Button from '../components/Button/Button'
import { startTraining, getTrainingStatus, cancelTraining } from '../services/api'
import './Training.css'

function Training() {
  const [formData, setFormData] = useState({
    symbol: 'MSFT',
    start_date: '2010-01-01',
    end_date: '2024-01-01',
    interval: '1d',
    timesteps: 100000
  })
  const [trainingStatus, setTrainingStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: isNaN(value) ? value : Number(value)
    }))
  }

  const handleStartTraining = async () => {
    setLoading(true)
    setMessage('')
    try {
      const response = await startTraining(formData)
      setMessage('✓ Training started successfully!')
      setTrainingStatus('running')
      
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(progressInterval)
            setTrainingStatus('completed')
            return 100
          }
          return prev + Math.random() * 15
        })
      }, 2000)
    } catch (error) {
      setMessage('✗ Error starting training: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const handleCancelTraining = async () => {
    try {
      await cancelTraining()
      setTrainingStatus('cancelled')
      setProgress(0)
      setMessage('Training cancelled')
    } catch (error) {
      setMessage('Error cancelling training')
    }
  }

  return (
    <div className="training">
      <h2>Model Training</h2>

      <div className="training-grid">
        <Card title="Training Configuration">
          <form className="training-form">
            <div className="form-group">
              <label>Stock Symbol</label>
              <input
                type="text"
                name="symbol"
                value={formData.symbol}
                onChange={handleInputChange}
                placeholder="e.g., MSFT, AAPL"
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Start Date</label>
                <input
                  type="date"
                  name="start_date"
                  value={formData.start_date}
                  onChange={handleInputChange}
                />
              </div>
              <div className="form-group">
                <label>End Date</label>
                <input
                  type="date"
                  name="end_date"
                  value={formData.end_date}
                  onChange={handleInputChange}
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Interval</label>
                <select name="interval" value={formData.interval} onChange={handleInputChange}>
                  <option value="1d">1 Day</option>
                  <option value="1h">1 Hour</option>
                  <option value="15m">15 Minutes</option>
                </select>
              </div>
              <div className="form-group">
                <label>Timesteps</label>
                <input
                  type="number"
                  name="timesteps"
                  value={formData.timesteps}
                  onChange={handleInputChange}
                />
              </div>
            </div>

            <div className="button-group">
              <Button 
                variant="primary" 
                onClick={handleStartTraining}
                loading={loading}
                disabled={trainingStatus === 'running'}
              >
                Start Training
              </Button>
              {trainingStatus === 'running' && (
                <Button variant="danger" onClick={handleCancelTraining}>
                  Cancel Training
                </Button>
              )}
            </div>
          </form>
        </Card>

        <Card title="Training Status">
          <div className="status-container">
            <div className="status-indicator">
              <div className={`indicator ${trainingStatus}`}></div>
              <span>{trainingStatus.toUpperCase()}</span>
            </div>

            {trainingStatus === 'running' && (
              <div className="progress-container">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                </div>
                <span className="progress-text">{progress.toFixed(0)}%</span>
              </div>
            )}

            {message && (
              <div className={`message ${message.includes('✓') ? 'success' : 'error'}`}>
                {message}
              </div>
            )}

            <div className="status-info">
              <div className="info-item">
                <span className="label">Symbol:</span>
                <span className="value">{formData.symbol}</span>
              </div>
              <div className="info-item">
                <span className="label">Period:</span>
                <span className="value">{formData.start_date} to {formData.end_date}</span>
              </div>
              <div className="info-item">
                <span className="label">Timesteps:</span>
                <span className="value">{formData.timesteps.toLocaleString()}</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Training Info">
        <div className="info-box">
          <h4>How LSTM-RL Training Works:</h4>
          <ul>
            <li><strong>LSTM Layer:</strong> Captures temporal dependencies in price sequences</li>
            <li><strong>Reinforcement Learning:</strong> Optimizes trading actions based on rewards</li>
            <li><strong>Feature Engineering:</strong> Automatic extraction of technical indicators</li>
            <li><strong>Portfolio Management:</strong> Balances risk and returns</li>
          </ul>
        </div>
      </Card>
    </div>
  )
}

export default Training
