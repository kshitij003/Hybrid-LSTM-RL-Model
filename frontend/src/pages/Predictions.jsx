import { useState, useEffect } from 'react'
import { predict, getPreferences, getPortfolioById, forceRebalance } from '../services/api'
import toast from 'react-hot-toast'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const PIE_COLORS = ['#00d4ff','#7c3aed','#10b981','#f59e0b','#ef4444','#6366f1','#ec4899','#14b8a6']
const FEE_RATE   = 0.001  // 0.1%

const fmtCurr = (n) => `₹${Number(n).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`

export default function Predictions() {
  const [prefs,     setPrefs]     = useState({ stocks: [] })
  const [holdings,  setHoldings]  = useState({})
  const [cash,      setCash]      = useState(100000)
  const [proposal,  setProposal]  = useState(null)
  const [ledger,    setLedger]    = useState([])
  const [loading,   setLoading]   = useState(false)
  const [executing, setExecuting] = useState(false)

  useEffect(() => {
    // Fetch real portfolio from DB (ID 1 for demo)
    getPortfolioById(1).then(r => {
      const p = r.data
      setCash(p.currentCashBalance)
      const h = {}
      p.holdings.forEach(item => {
        const price = item.stock.currentPrice || 100.0
        h[item.stock.ticker] = { 
          value: item.quantity * price, 
          qty: item.quantity 
        }
      })
      setHoldings(h)
      setPrefs({ stocks: p.holdings.map(item => item.stock.ticker) })
    }).catch(() => {
      // Fallback if DB is empty
      getPreferences().then(r => {
        const stocks = r.data.stocks || []
        setPrefs({ stocks })
      })
    })
  }, [])

  const totalValue = cash + Object.values(holdings).reduce((s,v)=>s+v, 0)

  const currentPieData = [
    ...Object.entries(holdings).map(([k,v]) => ({ name: k.replace('.NS',''), value: v })),
    { name: 'CASH', value: cash }
  ].filter(d => d.value > 0)

  const runPredict = async () => {
    if (prefs.stocks.length === 0) { toast.error('No stocks in portfolio.'); return }
    setLoading(true)
    try {
      const holdingsMap = {}
      Object.entries(holdings).forEach(([t, h]) => { holdingsMap[t] = h.qty })
      
      const res = await predict({ 
        currentCash: cash, 
        currentHoldings: holdingsMap, 
        tickers: prefs.stocks 
      })
      const weights = res.data.targetWeights || {}
      const rows = Object.entries(weights)
        .filter(([k]) => k !== 'CASH')
        .map(([asset, suggestedW]) => {
          const currV   = holdings[asset]?.value || 0
          const currW   = totalValue > 0 ? (currV / totalValue * 100) : 0
          const sugW    = suggestedW * 100
          return { asset, currW: currW.toFixed(1), sugW: sugW.toFixed(1), change: (sugW - currW).toFixed(1) }
        })
      rows.push({
        asset: 'CASH',
        currW:   (cash / totalValue * 100).toFixed(1),
        sugW:    ((weights.CASH || 0) * 100).toFixed(1),
        change:  (((weights.CASH||0)*100) - (cash/totalValue*100)).toFixed(1),
      })
      setProposal({ rows, weights, confidence: res.data.confidenceScore, modelVersion: res.data.modelVersion })
      toast.success('AI rebalance proposal generated!')
    } catch (err) {
      toast.error(err?.response?.data?.error?.message || 'Prediction failed — is the model trained?')
    } finally { setLoading(false) }
  }

  const executeRebalance = async () => {
    if (!proposal) return
    setExecuting(true)
    try {
      // Trigger real rebalance in Spring Boot
      const res = await forceRebalance(1)
      toast.success(res.data)
      
      // Refresh portfolio state
      const pRes = await getPortfolioById(1)
      const p = pRes.data
      setCash(p.currentCashBalance)
      const h = {}
      p.holdings.forEach(item => { h[item.stock.ticker] = item.quantity * 100 }) // placeholder price
      setHoldings(h)
      setProposal(null)
    } catch (err) {
      toast.error('DB rebalance execution failed.')
    } finally {
      setExecuting(false)
    }
  }

  return (
    <main className="page fade-in">
      <div className="page-header">
        <h1 className="page-title">Dashboard &amp; AI Rebalance Proposal</h1>
        <p className="page-subtitle">PPO agent suggests optimal weights based on LSTM signals</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 260px', gap: '1rem', marginBottom: '1.5rem' }}>
        {/* Current positions pie */}
        <div className="card">
          <div className="card-title">Current Positions</div>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie data={currentPieData} cx="50%" cy="50%" innerRadius={45} outerRadius={70}
                   dataKey="value" strokeWidth={0}>
                {currentPieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
              </Pie>
              <Tooltip formatter={(v) => fmtCurr(v)} contentStyle={{
                background:'#1e2538', border:'1px solid rgba(0,212,255,0.2)', borderRadius:8, fontSize:'0.78rem' }} />
            </PieChart>
          </ResponsiveContainer>
          {currentPieData.slice(0,4).map((d,i) => (
            <div key={d.name} className="legend-item">
              <div className="legend-dot" style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
              <span style={{ fontSize: '0.78rem' }}>{d.name}</span>
              <span style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                {(d.value / totalValue * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>

        {/* Proposal */}
        <div className="card">
          <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>AI Rebalance Proposal (PPO Agent)</span>
            {proposal && <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>
              Confidence: <strong style={{ color: 'var(--cyan)' }}>{(proposal.confidence*100).toFixed(0)}%</strong>
            </span>}
          </div>

          {!proposal ? (
            <div style={{ textAlign: 'center', padding: '1.5rem 0' }}>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.88rem' }}>
                Click below to get the PPO agent's suggested allocation.
              </p>
              <button className="btn btn-primary" onClick={runPredict} disabled={loading}>
                {loading ? '⏳ Fetching…' : '🤖 Get AI Proposal'}
              </button>
            </div>
          ) : (
            <>
              <div className="table-wrap" style={{ marginBottom: '1rem' }}>
                <table>
                  <thead><tr>
                    <th>Asset</th><th>Current %</th><th>PPO Suggested %</th><th>Change %</th>
                  </tr></thead>
                  <tbody>
                    {proposal.rows.map(r => (
                      <tr key={r.asset}>
                        <td style={{ fontWeight: 600 }}>{(r.asset || '').replace('.NS','')}</td>
                        <td>{r.currW}%</td>
                        <td style={{ color: 'var(--cyan)' }}>{r.sugW}%</td>
                        <td className={parseFloat(r.change)>0?'change-pos':parseFloat(r.change)<0?'change-neg':'change-zero'}>
                          {parseFloat(r.change)>0?'+':''}{r.change}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button className="btn btn-primary" style={{ flex: 1 }}
                  onClick={executeRebalance} disabled={executing}>
                  {executing ? '⏳ Executing…' : '⚡ Execute AI Rebalance'}
                </button>
                <button className="btn btn-ghost" onClick={() => setProposal(null)}>Cancel</button>
              </div>
            </>
          )}
        </div>

        <div className="card">
          <div className="card-title">Portfolio State</div>
          <div className="summary-grid">
            <div className="summary-row">
              <span className="summary-key">Total Value</span>
              <span className="summary-value gold">{fmtCurr(totalValue)}</span>
            </div>
            <div className="summary-row">
              <span className="summary-key">Cash</span>
              <span className="summary-value">{fmtCurr(cash)}</span>
            </div>
            {Object.entries(holdings).map(([k,v]) => (
              <div key={k} className="summary-row">
                <span className="summary-key">{(k || '').replace('.NS','')} ({v?.qty || 0})</span>
                <span className="summary-value">{fmtCurr(v?.value || 0)}</span>
              </div>
            ))}
          </div>
          <button className="btn btn-secondary" style={{ width:'100%', marginTop:'0.75rem' }}
            onClick={runPredict} disabled={loading}>
            {loading ? '⏳…' : '🔄 Refresh Proposal'}
          </button>
        </div>
      </div>

      <div className="card">
        <div className="section-header">
          <div className="section-title">Transaction Log &amp; Fee Ledger</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>* 0.1% transaction fee applied</span>
        </div>
        {ledger.length === 0 ? (
          <div className="empty-state" style={{ padding: '1.5rem' }}>
            <p>No transactions yet. Execute a rebalance to populate the ledger.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr>
                <th>Date / Time</th><th>Action</th><th>Asset</th>
                <th>Amount (Gross)</th><th>Net Value</th><th>Fee (0.1%)</th>
              </tr></thead>
              <tbody>
                {ledger.map((e, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{e.datetime}</td>
                    <td>
                      <span className={`badge ${e.action==='Buy'?'badge-positive':'badge-negative'}`}>
                        {e.action}
                      </span>
                    </td>
                    <td style={{ fontWeight: 600 }}>{(e.asset || '').replace('.NS','')}</td>
                    <td>{fmtCurr(e.grossAmt)}</td>
                    <td>{fmtCurr(e.netAmt)}</td>
                    <td style={{ color: 'var(--red)', fontSize: '0.82rem' }}>-{fmtCurr(e.fee)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  )
}
