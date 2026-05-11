import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { getDashboardMetrics, listModels, listTrainingJobs, getPreferences, syncMarketData } from '../services/api'
import toast from 'react-hot-toast'

const CHART_COLORS = ['#00d4ff','#7c3aed','#10b981','#f59e0b','#ef4444']

function MiniBarChart({ data }) {
  if (!data || data.length === 0) return null
  const max = Math.max(...data.map(d => Math.abs(d.value)), 0.01)
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 60, padding: '0.5rem 0' }}>
      {data.map((d, i) => (
        <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
          <div
            style={{
              width: '100%',
              height: `${Math.max((Math.abs(d.value) / max) * 50, 4)}px`,
              background: d.value >= 0 ? 'var(--cyan)' : 'var(--red)',
              borderRadius: '2px 2px 0 0',
              opacity: 0.8,
              transition: 'height 0.3s ease',
            }}
          />
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{d.label}</span>
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [metrics, setMetrics] = useState(null)
  const [models, setModels] = useState([])
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      const [metricsRes, modelsRes, jobsRes] = await Promise.allSettled([
        getDashboardMetrics(),
        listModels(),
        listTrainingJobs()
      ])
      
      if (metricsRes.status === 'fulfilled') setMetrics(metricsRes.value.data)
      if (modelsRes.status === 'fulfilled') setModels(modelsRes.value.data.models || [])
      if (jobsRes.status === 'fulfilled') setJobs((jobsRes.value.data.jobs || []).slice(0, 5))
    } catch (e) {
      toast.error('Could not load dashboard data')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSync = async () => {
    toast.info('Syncing market data from Yahoo Finance...')
    try {
      await syncMarketData()
      toast.success('Market data synced!')
      load() // Reload to show updated values
    } catch (err) {
      toast.error('Sync failed')
    }
  }

  useEffect(() => { load() }, [load])

  const lastJob = jobs[0]
  const activeModel = models.find(m => m.isActive)
  const portfolioVal = metrics?.portfolioValue || 150000
  const dayChange = metrics?.dayChange || 0
  const aiConfidence = metrics?.aiConfidence || 0.88

  const chartData = [
    { label: 'Mon', value: 2.1 },
    { label: 'Tue', value: -0.8 },
    { label: 'Wed', value: 3.4 },
    { label: 'Thu', value: 1.2 },
    { label: 'Fri', value: -1.5 },
    { label: 'Sat', value: 0.6 },
    { label: 'Sun', value: 2.8 },
  ]

  return (
    <main className="page fade-in">
      <div className="welcome-banner">
        <h2>Welcome to Hybrid LSTM-RL Trading System</h2>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.88rem' }}>
          AI-powered portfolio rebalancing using LSTM feature extraction + PPO reinforcement learning.
          {metrics?.activeStocks?.length > 0 && ` Active stocks: ${metrics.activeStocks.map(s => s.ticker).join(', ')}`}
        </p>
        <button 
          onClick={handleSync}
          className="button-primary"
          style={{ marginTop: '1rem', background: 'var(--blue-gradient)', padding: '0.5rem 1rem', fontSize: '0.8rem' }}
        >
          🔄 Sync Market Data
        </button>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Training Status</div>
          <div className="stat-value" style={{ fontSize: '1.5rem' }}>
            {lastJob?.status === 'IN_PROGRESS' ? 'TRAINING' : 'IDLE'}
          </div>
          <div className="stat-meta">Model Training</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">AI Confidence</div>
          <div className="stat-value" style={{ color: aiConfidence > 0.8 ? 'var(--green)' : 'var(--yellow)' }}>
            {(aiConfidence * 100).toFixed(0)}%
          </div>
          <div className="stat-meta">Portfolio Logic</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Day Change</div>
          <div className={`stat-value ${dayChange >= 0 ? 'green' : 'red'}`}>
            {dayChange >= 0 ? '+' : ''}{dayChange.toFixed(2)}%
          </div>
          <div className="stat-meta">Market Performance</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Portfolio Value</div>
          <div className="stat-value gold">₹{portfolioVal.toLocaleString()}</div>
          <div className="stat-meta">Current Equity (DB)</div>
        </div>
      </div>

      {/* Quick Start */}
      <div className="card section">
        <div className="card-title">Quick Start</div>
        <div className="quick-actions">
          <button className="btn btn-primary" onClick={() => navigate('/training')}>
            🚀 Start Training
          </button>
          <button className="btn btn-secondary" onClick={() => navigate('/backtesting')}>
            📊 Run Backtest
          </button>
          <button className="btn btn-secondary" onClick={() => navigate('/predictions')}>
            🤖 Make Prediction
          </button>
          <button className="btn btn-ghost" onClick={() => navigate('/portfolios')}>
            ⚙️ Configure Portfolio
          </button>
        </div>
      </div>

      {/* Bottom row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        {/* Recent backtest chart */}
        <div className="card">
          <div className="card-title">Weekly P&L Simulation (%)</div>
          <MiniBarChart data={chartData} />
        </div>

        {/* Recent jobs */}
        <div className="card">
          <div className="card-title">Recent Training Jobs</div>
          {loading ? <div className="spinner" /> :
           jobs.length === 0 ? (
             <div className="empty-state" style={{ padding: '1rem' }}>
               <p>No training jobs yet. <br/>
                 <button className="btn btn-primary btn-sm" style={{ marginTop: '0.5rem' }}
                   onClick={() => navigate('/training')}>Start Training</button>
               </p>
             </div>
           ) : (
             <div className="table-wrap">
               <table>
                 <thead><tr>
                   <th>ID</th><th>Status</th><th>Stocks</th>
                 </tr></thead>
                 <tbody>
                   {jobs.map(j => (
                     <tr key={j.trainingId}>
                       <td style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>{j.trainingId}</td>
                       <td>
                         <span className={`badge badge-${
                           j.status === 'COMPLETED' ? 'done' :
                           j.status === 'IN_PROGRESS' ? 'running' : 'idle'}`}>
                           {j.status}
                         </span>
                       </td>
                       <td style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                         {(j.stocks || []).slice(0,3).join(', ')}
                         {(j.stocks||[]).length > 3 && ` +${j.stocks.length-3}`}
                       </td>
                     </tr>
                   ))}
                 </tbody>
               </table>
             </div>
           )}
        </div>
      </div>

      {/* Models list */}
      {models.length > 0 && (
        <div className="card section" style={{ marginTop: '1rem' }}>
          <div className="card-title">Available Models</div>
          <div className="table-wrap">
            <table>
              <thead><tr>
                <th>Model ID</th><th>Status</th><th>Trained On</th><th>Size</th>
              </tr></thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.modelId}>
                    <td style={{ fontWeight: 600 }}>{m.modelId}</td>
                    <td>
                      {m.isActive
                        ? <span className="badge badge-running">Active</span>
                        : <span className="badge badge-idle">Inactive</span>}
                    </td>
                    <td style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                      {m.trainedOn ? new Date(m.trainedOn).toLocaleDateString() : '—'}
                    </td>
                    <td style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                      {m.sizeBytes ? `${(m.sizeBytes/1024).toFixed(0)} KB` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </main>
  )
}
