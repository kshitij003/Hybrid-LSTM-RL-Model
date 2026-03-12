import React, { useState } from 'react'
import Card from '../components/Card/Card'
import Button from '../components/Button/Button'
import { makePrediction, batchPredict } from '../services/api'
import './Predictions.css'

function Predictions() {
  const [formData, setFormData] = useState({
    symbol: 'MSFT',
    days_ahead: 1
  })
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [batchSymbols, setBatchSymbols] = useState('MSFT, AAPL, GOOGL')
  const [batchPredictions, setBatchPredictions] = useState([])

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: isNaN(value) ? value : Number(value)
    }))
  }

  const handleMakePrediction = async () => {
    setLoading(true)
    try {
      const response = await makePrediction(formData)
      setPrediction(response.data)
    } catch (error) {
      console.error('Error making prediction:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleBatchPredict = async () => {
    setLoading(true)
    try {
      const symbols = batchSymbols.split(',').map(s => s.trim())
      const response = await batchPredict(symbols)
      setBatchPredictions(response.data.predictions || [])
    } catch (error) {
      console.error('Error batch predicting:', error)
    } finally {
      setLoading(false)
    }
  }

  const getRecommendationColor = (rec) => {
    if (rec === 'BUY') return 'green'
    if (rec === 'SELL') return 'red'
    return 'yellow'
  }

  return (
    <div className="predictions">
      <h2>Price Predictions & Trading Signals</h2>

      <div className="predictions-grid">
        <Card title="Single Stock Prediction">
          <form className="prediction-form">
            <div className="form-group">
              <label>Stock Symbol</label>
              <input
                type="text"
                name="symbol"
                value={formData.symbol}
                onChange={handleInputChange}
                placeholder="e.g., MSFT"
              />
            </div>

            <div className="form-group">
              <label>Days Ahead</label>
              <select name="days_ahead" value={formData.days_ahead} onChange={handleInputChange}>
                <option value={1}>1 Day</option>
                <option value={3}>3 Days</option>
                <option value={7}>1 Week</option>
                <option value={30}>1 Month</option>
              </select>
            </div>

            <Button 
              variant="primary" 
              onClick={handleMakePrediction}
              loading={loading}
            >
              Predict
            </Button>
          </form>
        </Card>

        {prediction && (
          <Card title="Prediction Result">
            <div className="prediction-result">
              <div className="symbol-header">
                <h3>{prediction.symbol}</h3>
                <div className={`recommendation ${prediction.recommendation.toLowerCase()}`}>
                  {prediction.recommendation}
                </div>
              </div>

              <div className="prediction-details">
                <div className="detail-item">
                  <span className="label">Current Price</span>
                  <span className="value">${prediction.current_price?.toFixed(2)}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Predicted Price</span>
                  <span className="value predicted">${prediction.predicted_price?.toFixed(2)}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Expected Change</span>
                  <span className={`value ${prediction.price_change_percent > 0 ? 'green' : 'red'}`}>
                    {prediction.price_change_percent?.toFixed(2)}%
                  </span>
                </div>
                <div className="detail-item">
                  <span className="label">Confidence</span>
                  <span className="value">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              {prediction.factors && (
                <div className="factors">
                  <h4>Market Factors</h4>
                  <div className="factor-list">
                    <div className="factor">
                      <span className="factor-name">Momentum</span>
                      <span className={`factor-value ${prediction.factors.momentum?.toLowerCase()}`}>
                        {prediction.factors.momentum}
                      </span>
                    </div>
                    <div className="factor">
                      <span className="factor-name">Volatility</span>
                      <span className="factor-value">{prediction.factors.volatility}</span>
                    </div>
                    <div className="factor">
                      <span className="factor-name">Trend</span>
                      <span className={`factor-value ${prediction.factors.trend?.toLowerCase()}`}>
                        {prediction.factors.trend}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}
      </div>

      <Card title="Batch Predictions">
        <div className="batch-section">
          <div className="batch-input">
            <label>Symbols (comma separated)</label>
            <input
              type="text"
              value={batchSymbols}
              onChange={(e) => setBatchSymbols(e.target.value)}
              placeholder="e.g., MSFT, AAPL, GOOGL"
            />
          </div>

          <Button variant="secondary" onClick={handleBatchPredict} loading={loading}>
            Predict All
          </Button>

          {batchPredictions.length > 0 && (
            <div className="batch-results">
              <div className="results-grid">
                {batchPredictions.map(pred => (
                  <div key={pred.symbol} className="batch-result-card">
                    <h4>{pred.symbol}</h4>
                    <div className={`label ${pred.recommendation.toLowerCase()}`}>
                      {pred.recommendation}
                    </div>
                    <div className="batch-details">
                      <div className="batch-detail">
                        <span className="label">Price</span>
                        <span className="value">${pred.predicted_price?.toFixed(2)}</span>
                      </div>
                      <div className="batch-detail">
                        <span className="label">Confidence</span>
                        <span className="value">{(pred.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>

      <Card title="How Predictions Work">
        <div className="info-box">
          <h4>Prediction Model</h4>
          <ul>
            <li><strong>LSTM Analysis:</strong> Analyzes historical price sequences</li>
            <li><strong>Pattern Recognition:</strong> Identifies repeating market patterns</li>
            <li><strong>Technical Indicators:</strong> RSI, MACD, Bollinger Bands analysis</li>
            <li><strong>RL Strategy:</strong> Reinforcement learning optimal action selection</li>
            <li><strong>Confidence Score:</strong> Model certainty in prediction</li>
          </ul>
        </div>
      </Card>
    </div>
  )
}

export default Predictions
