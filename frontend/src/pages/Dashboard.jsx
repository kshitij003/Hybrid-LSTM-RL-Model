import React, { useState, useEffect } from 'react'
import Card from '../components/Card/Card'
import { checkHealth, getTrainingStatus, getBacktestHistory, getPredictionHistory } from '../services/api'
import './Dashboard.css'

function Dashboard() {
  const [stats, setStats] = useState({
    trainingStatus: 'idle',
    lastBacktest: null,
    predictions: 0
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const [statusRes, backtestRes] = await Promise.all([
          getTrainingStatus(),
          getBacktestHistory()
        ])
        
        setStats({
          trainingStatus: statusRes.data.status,
          lastBacktest: backtestRes.data.backtests[0] || null,
          predictions: 42
        })
      } catch (error) {
        console.error('Error fetching dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchDashboardData()
  }, [])

  return (
    <div className="dashboard">
      <h2>Welcome to Hybrid LSTM-RL Trading System</h2>
      
      <div className="stats-grid">
        <Card title="Training Status" className="stat-card">
          <div className="stat-value">{stats.trainingStatus.toUpperCase()}</div>
          <p className="stat-label">Model Trained</p>
        </Card>

        <Card title="Last Backtest" className="stat-card">
          {stats.lastBacktest ? (
            <>
              <div className="stat-value">{stats.lastBacktest.win_rate.toFixed(0)}%</div>
              <p className="stat-label">Win Rate</p>
              <small className="stat-date">{stats.lastBacktest.timestamp}</small>
            </>
          ) : (
            <p className="stat-label">No backtests yet</p>
          )}
        </Card>

        <Card title="Total Predictions" className="stat-card">
          <div className="stat-value">{stats.predictions}</div>
          <p className="stat-label">Completed</p>
        </Card>

        <Card title="Portfolio Value" className="stat-card">
          <div className="stat-value green">$150,000</div>
          <p className="stat-label">Current Equity</p>
        </Card>
      </div>

      {stats.lastBacktest && (
        <Card title="Latest Backtest Results">
          <div className="results-grid">
            <div className="result-item">
              <span className="result-label">Final Equity:</span>
              <span className="result-value">${stats.lastBacktest.final_equity.toLocaleString()}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Total Return:</span>
              <span className="result-value green">${stats.lastBacktest.total_return.toLocaleString()}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Win Rate:</span>
              <span className="result-value">{(stats.lastBacktest.win_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="result-item">
              <span className="result-label">Sharpe Ratio:</span>
              <span className="result-value">{stats.lastBacktest.sharpe_ratio?.toFixed(2) || 'N/A'}</span>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}

export default Dashboard
