import axios from 'axios'

// Point to Spring Boot Backend (Persistence Layer)
const api = axios.create({
  baseURL: 'http://localhost:8080',
  headers: { 'Content-Type': 'application/json' },
  timeout: 120000, // Longer timeout for training proxy
})

let authToken = null
export const setToken = (token) => { authToken = token }

api.interceptors.request.use(config => {
  if (authToken) {
    config.headers.Authorization = `Bearer ${authToken}`
  }
  return config
})

api.interceptors.response.use(
  res => res,
  err => {
    console.error('API Error:', err?.response?.data || err.message)
    return Promise.reject(err)
  }
)

// ── Dashboard (Spring Boot Real Data) ─────────────────────────────────────────
export const getDashboardMetrics = () => api.get('/api/dashboard/metrics')

// ── Health (Proxied via Spring Boot) ──────────────────────────────────────────
export const checkHealth = () => api.get('/api/ml/health')

// ── Training (Proxied via Spring Boot) ────────────────────────────────────────
export const startTraining      = (d)  => api.post('/api/ml/train/full', d)
export const getTrainingStatus  = (id) => api.get(`/api/ml/training-status/${id}`)
export const listTrainingJobs   = ()   => api.get('/api/ml/training-jobs')
export const listModels         = ()   => api.get('/api/ml/models')
export const quickUpdate        = (d)  => api.post('/api/ml/train/quick-update', d)

// ── Portfolio Preferences (Proxied via Spring Boot) ───────────────────────────
export const getPreferences     = ()   => api.get('/api/ml/portfolio/preferences')
export const savePreferences    = (d)  => api.post('/api/ml/portfolio/preferences', d)
export const getSupportedStocks = (q)  => api.get('/api/ml/portfolio/supported-stocks', { params: q ? { q } : {} })
export const trainCustom        = (d)  => api.post('/api/ml/portfolio/train-custom', d)

// ── Inference / Predictions (Proxied via Spring Boot) ─────────────────────────
export const predict = (d) => api.post('/api/ml/predict', d)

// ── Backtesting (Proxied via Spring Boot) ─────────────────────────────────────
export const runBacktest = (d) => api.post('/api/ml/backtest/run', d)

// ── News (Proxied via Spring Boot) ────────────────────────────────────────────
export const getNews = () => api.get('/api/ml/news/headlines')

// ── Portfolio Management (Direct Spring Boot DB) ──────────────────────────────
export const getPortfolioById = (id) => api.get(`/api/portfolio/${id}`)
export const forceRebalance   = (id) => api.post(`/api/portfolio/${id}/rebalance`)
export const syncMarketData   = ()   => api.post('/api/portfolio/sync-data')

export default api
