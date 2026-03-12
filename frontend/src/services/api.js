import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Health Check
export const checkHealth = () => api.get('/health')

// Training
export const startTraining = (config) => api.post('/train', config)
export const getTrainingStatus = () => api.get('/training-status')
export const cancelTraining = () => api.post('/train/cancel')

// Backtesting
export const runBacktest = (config) => api.post('/backtest', config)
export const getBacktestHistory = () => api.get('/backtest/history')
export const getBacktestDetails = (backtestId) => api.get(`/backtest/${backtestId}/details`)

// Predictions
export const makePrediction = (request) => api.post('/predict', request)
export const batchPredict = (symbols) => api.post('/predict/batch', { symbols })
export const getPredictionHistory = (symbol, limit = 10) => 
  api.get('/predict/history', { params: { symbol, limit } })

export default api
