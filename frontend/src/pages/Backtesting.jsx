import { useState } from 'react'
import { runBacktest, getPreferences } from '../services/api'
import toast from 'react-hot-toast'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#1e2538', border: '1px solid rgba(0,212,255,0.2)',
                  padding: '0.6rem 0.9rem', borderRadius: 8, fontSize: '0.8rem' }}>
      <p style={{ color: 'var(--cyan)', fontWeight: 700 }}>{label}</p>
      <p style={{ color: payload[0].value > 0 ? 'var(--green)' : 'var(--red)' }}>
        Return: {payload[0].value?.toFixed(2)}%
      </p>
    </div>
  )
}

export default function Backtesting() {
  const [form,    setForm]    = useState({
    startDate:    '2023-01-01',
    endDate:      '2024-12-31',
    initialCash:  100000,
  })
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)

  // We simulate a backtest by running predict() across sample data
  const handleRunBacktest = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await runBacktest({
        startDate: form.startDate,
        endDate: form.endDate,
        initialBalance: Number(form.initialCash)
      })

      setResults({
        targetWeights: {}, // backend results don't have weights in this simple format yet
        confidence: 0.85,
        chartData: res.data.chartData,
        totalReturn: res.data.totalReturn,
        winRate: res.data.winRate,
        sharpe: res.data.sharpe,
        stocks: [], // can be populated if needed
      })
      toast.success('Real historical backtest complete!')
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Backtest failed — is the model trained?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page fade-in">
      <div className="page-header">
        <h1 className="page-title">Backtesting</h1>
        <p className="page-subtitle">Simulate AI portfolio performance over historical data</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Form */}
        <div className="card">
          <div className="card-title">Backtest Parameters</div>
          <form onSubmit={handleRunBacktest}>
            <div className="form-group">
              <label className="form-label">Start Date</label>
              <input className="form-input" type="date" value={form.startDate}
                onChange={e => setForm(f=>({...f,startDate:e.target.value}))} />
            </div>
            <div className="form-group">
              <label className="form-label">End Date</label>
              <input className="form-input" type="date" value={form.endDate}
                onChange={e => setForm(f=>({...f,endDate:e.target.value}))} />
            </div>
            <div className="form-group">
              <label className="form-label">Initial Capital (₹)</label>
              <input className="form-input" type="number" min={1000} value={form.initialCash}
                onChange={e => setForm(f=>({...f,initialCash:e.target.value}))} />
            </div>
            <button className="btn btn-primary" style={{ width: '100%' }}
              type="submit" disabled={loading}>
              {loading ? '⏳ Running…' : '▶ Run Backtest'}
            </button>
          </form>

          {results && (
            <div style={{ marginTop: '1.25rem' }}>
              <div className="card-title">Summary</div>
              <div className="summary-grid">
                <div className="summary-row">
                  <span className="summary-key">Total Return</span>
                  <span className="summary-value" style={{ color: results.totalReturn >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {results.totalReturn.toFixed(2)}%
                  </span>
                </div>
                <div className="summary-row">
                  <span className="summary-key">Win Rate</span>
                  <span className="summary-value">{results.winRate}%</span>
                </div>
                <div className="summary-row">
                  <span className="summary-key">Sharpe Ratio</span>
                  <span className="summary-value">{results.sharpe}</span>
                </div>
                <div className="summary-row">
                  <span className="summary-key">Confidence</span>
                  <span className="summary-value">{(results.confidence*100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Results */}
        <div>
          {!results && !loading && (
            <div className="card">
              <div className="empty-state">
                <div className="empty-icon">📈</div>
                <p>Run a backtest to see monthly returns, target weights, and performance metrics.</p>
              </div>
            </div>
          )}

          {loading && <div className="card"><div className="spinner" /></div>}

          {results && (
            <>
              <div className="card section">
                <div className="card-title">Monthly Returns (%)</div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={results.chartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                    <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="return" radius={[4,4,0,0]}>
                      {results.chartData.map((d,i) => (
                        <Cell key={i} fill={d.return >= 0 ? '#00d4ff' : '#ef4444'} opacity={0.85} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <div className="card-title">AI Suggested Weights</div>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Asset</th><th>Suggested Weight (%)</th></tr></thead>
                    <tbody>
                      {Object.entries(results.targetWeights).map(([k,v]) => (
                        <tr key={k}>
                          <td style={{ fontWeight: 600 }}>{k}</td>
                          <td>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              <div style={{
                                width: `${(v*100).toFixed(0)}%`, maxWidth: '100%',
                                height: 6, background: 'var(--cyan)',
                                borderRadius: 3, opacity: 0.7, minWidth: 4 }} />
                              <span>{(v*100).toFixed(1)}%</span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </main>
  )
}
