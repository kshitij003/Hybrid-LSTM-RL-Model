import React, { useState, useEffect } from 'react'
import Card from '../components/Card/Card'
import Button from '../components/Button/Button'
import { runBacktest, getBacktestHistory, getBacktestDetails } from '../services/api'
import './Backtesting.css'

function Backtesting() {
  const [formData, setFormData] = useState({
    symbol: 'MSFT',
    start_date: '2020-01-01',
    end_date: '2024-01-01',
    model_path: null
  })
  const [backtestHistory, setBacktestHistory] = useState([])
  const [selectedBacktest, setSelectedBacktest] = useState(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetchBacktestHistory()
  }, [])

  const fetchBacktestHistory = async () => {
    try {
      const response = await getBacktestHistory()
      setBacktestHistory(response.data.backtests || [])
    } catch (error) {
      console.error('Error fetching backtest history:', error)
    }
  }

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleRunBacktest = async () => {
    setLoading(true)
    setMessage('')
    try {
      const response = await runBacktest(formData)
      setMessage('✓ Backtest completed successfully!')
      fetchBacktestHistory()
      
      if (response.data.results) {
        setSelectedBacktest(response.data.results)
      }
    } catch (error) {
      setMessage('✗ Error running backtest: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="backtesting">
      <h2>Backtesting</h2>

      <div className="backtest-grid">
        <Card title="Backtest Configuration">
          <form className="backtest-form">
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

            <Button 
              variant="primary" 
              onClick={handleRunBacktest}
              loading={loading}
            >
              Run Backtest
            </Button>

            {message && (
              <div className={`message ${message.includes('✓') ? 'success' : 'error'}`}>
                {message}
              </div>
            )}
          </form>
        </Card>

        {selectedBacktest && (
          <Card title="Backtest Results">
            <div className="results-container">
              <div className="result-box">
                <span className="result-label">Final Equity</span>
                <span className="result-value green">${selectedBacktest.final_equity?.toLocaleString()}</span>
              </div>
              <div className="result-box">
                <span className="result-label">Total Return</span>
                <span className="result-value green">${selectedBacktest.total_return?.toLocaleString()}</span>
              </div>
              <div className="result-box">
                <span className="result-label">Win Rate</span>
                <span className="result-value">{((selectedBacktest.win_rate || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="result-box">
                <span className="result-label">Max Drawdown</span>
                <span className="result-value danger">-${selectedBacktest.max_drawdown?.toLocaleString()}</span>
              </div>
              <div className="result-box">
                <span className="result-label">Sharpe Ratio</span>
                <span className="result-value">{(selectedBacktest.sharpe_ratio || 0).toFixed(2)}</span>
              </div>
              <div className="result-box">
                <span className="result-label">Total Trades</span>
                <span className="result-value">{selectedBacktest.total_trades}</span>
              </div>
            </div>
          </Card>
        )}
      </div>

      <Card title="Backtest History">
        <div className="history-table">
          {backtestHistory.length > 0 ? (
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Date</th>
                  <th>Symbol</th>
                  <th>Final Equity</th>
                  <th>Win Rate</th>
                  <th>Return</th>
                </tr>
              </thead>
              <tbody>
                {backtestHistory.map(backtest => (
                  <tr key={backtest.id}>
                    <td>#{backtest.id}</td>
                    <td>{backtest.timestamp}</td>
                    <td><strong>{backtest.symbol}</strong></td>
                    <td>${backtest.final_equity?.toLocaleString()}</td>
                    <td>{(backtest.win_rate * 100).toFixed(1)}%</td>
                    <td className="positive">${backtest.total_return?.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="no-data">No backtest history yet. Run a backtest to see results.</p>
          )}
        </div>
      </Card>
    </div>
  )
}

export default Backtesting
