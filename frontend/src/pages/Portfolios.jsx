import { useState, useEffect, useRef } from 'react'
import { getPreferences, savePreferences, getSupportedStocks, trainCustom, getTrainingStatus } from '../services/api'
import toast from 'react-hot-toast'

const FREQ_OPTIONS  = ['Daily','Weekly','Monthly','Quarterly']
const VALUE_OPTIONS = ['Fixed','Percent','Dynamic']
const UNIT_OPTIONS  = ['Percent','INR']

const STAGE_LABELS = {
  INITIALIZING:'Initializing…', DATA_DOWNLOAD:'Downloading data…',
  LSTM_TRAINING:'Training LSTMs…', LOADING_LSTM:'Loading LSTMs…',
  BUILDING_ENV:'Building env…', PPO_FINETUNING:'Fine-tuning PPO…', FINISHED:'Complete!',
}

export default function Portfolios() {
  const [stocks,    setStocks]   = useState([])
  const [search,    setSearch]   = useState('')
  const [results,   setResults]  = useState([])
  const [showDrop,  setShowDrop] = useState(false)
  const [freq,      setFreq]     = useState('Monthly')
  const [valType,   setValType]  = useState('Fixed')
  const [valUnit,   setValUnit]  = useState('Percent')
  const [saving,    setSaving]   = useState(false)
  const [trainJob,  setTrainJob] = useState(null)
  const [polling,   setPolling]  = useState(false)
  const searchRef  = useRef(null)
  const pollRef    = useRef(null)

  // Load prefs on mount
  useEffect(() => {
    getPreferences().then(r => {
      setStocks(r.data.stocks || [])
      setFreq(r.data.rebalanceFrequency || 'Monthly')
      setValType(r.data.initialValueType || 'Fixed')
      setValUnit(r.data.initialValueUnit || 'Percent')
    }).catch(() => {})
    return () => clearInterval(pollRef.current)
  }, [])

  // Search supported stocks
  useEffect(() => {
    if (!search.trim()) { setResults([]); setShowDrop(false); return }
    const t = setTimeout(async () => {
      try {
        const r = await getSupportedStocks(search)
        setResults((r.data.stocks || []).filter(s => !stocks.includes(s.symbol)).slice(0, 8))
        setShowDrop(true)
      } catch {}
    }, 300)
    return () => clearTimeout(t)
  }, [search, stocks])

  const addStock = (symbol) => {
    if (stocks.includes(symbol))  { toast.error('Already added'); return }
    if (stocks.length >= 10)      { toast.error('Max 10 stocks allowed'); return }
    setStocks(p => [...p, symbol])
    setSearch(''); setResults([]); setShowDrop(false)
  }

  const removeStock = (s) => setStocks(p => p.filter(x => x !== s))

  const handleSave = async () => {
    if (stocks.length === 0) { toast.error('Select at least 1 stock'); return }
    setSaving(true)
    try {
      await savePreferences({ stocks, rebalanceFrequency: freq, initialValueType: valType, initialValueUnit: valUnit })
      toast.success('Portfolio saved! Starting custom training…')
      // Auto-trigger training
      const res = await trainCustom({ stocks })
      const id  = res.data.trainingId
      setTrainJob({ trainingId: id, status: 'QUEUED', progress: { stage: 'INITIALIZING', percentComplete: 0 } })
      setPolling(true)
      pollRef.current = setInterval(async () => {
        try {
          const s = await getTrainingStatus(id)
          setTrainJob(s.data)
          if (['COMPLETED','FAILED'].includes(s.data.status)) {
            clearInterval(pollRef.current); setPolling(false)
            if (s.data.status === 'COMPLETED') toast.success('Model retrained on your new stocks! 🎉')
            else toast.error(`Training failed: ${s.data.error}`)
          }
        } catch { clearInterval(pollRef.current); setPolling(false) }
      }, 3000)
    } catch (err) {
      toast.error(err?.response?.data?.error?.message || 'Save failed')
    } finally { setSaving(false) }
  }

  const pct   = trainJob?.progress?.percentComplete || 0
  const stage = trainJob?.progress?.stage || 'INITIALIZING'

  const constraintColor = stocks.length === 0 ? 'var(--red)'
    : stocks.length <= 10 ? 'var(--green)' : 'var(--red)'

  return (
    <main className="page fade-in">
      <div className="page-header">
        <h1 className="page-title">Configure Portfolio</h1>
        <p className="page-subtitle">Select NSE stocks — the system will train dedicated LSTM models and fine-tune the PPO agent on your selection</p>
      </div>

      <div className="two-col">
        {/* Main panel */}
        <div>
          {/* Stock selection card */}
          <div className="card section">
            <div className="card-title">
              Portfolio Stock Selection
              <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: '0.5rem' }}>
                (Maximum 10 assets allowed)
              </span>
            </div>

            {/* Search row */}
            <div className="search-row">
              <div className="search-wrap" ref={searchRef}>
                <input
                  className="form-input"
                  placeholder="Search NSE ticker (e.g. RELIANCE, TCS, INFY)…"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  onFocus={() => results.length > 0 && setShowDrop(true)}
                  onBlur={() => setTimeout(() => setShowDrop(false), 150)}
                />
                {showDrop && results.length > 0 && (
                  <div className="search-results">
                    {results.map(s => (
                      <button key={s.symbol} className="search-item" onMouseDown={() => addStock(s.symbol)}>
                        <div>
                          <span className="search-item-symbol">{s.symbol.replace('.NS','')}</span>
                          <span style={{ marginLeft: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                            {s.name}
                          </span>
                        </div>
                        <span className="search-item-meta tag">{s.sector}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button className="btn btn-primary"
                onClick={() => { const s = search.trim().toUpperCase(); if (s) addStock(s.includes('.NS')?s:s+'.NS') }}
                disabled={!search.trim()}>
                + Add Stock
              </button>
            </div>

            {/* Asset count bar */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                {stocks.length} / 10 Assets
              </span>
              <div style={{ flex: 1 }}>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${(stocks.length / 10) * 100}%`,
                    background: stocks.length === 10 ? 'linear-gradient(90deg,var(--gold),var(--red))' :
                                'linear-gradient(90deg,var(--cyan),var(--purple))' }} />
                </div>
              </div>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {(stocks.length / 10 * 100).toFixed(0)}%
              </span>
            </div>

            {/* Chip list */}
            {stocks.length > 0 ? (
              <div className="chip-list">
                {stocks.map(s => (
                  <div key={s} className="chip">
                    {s.replace('.NS','')}
                    <button className="chip-remove" onClick={() => removeStock(s)}>✕</button>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '0.5rem 0' }}>
                Search and add NSE stocks above.
              </p>
            )}
          </div>

          {/* Settings */}
          <div className="card section">
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Rebalance Frequency</label>
                <select className="form-select" value={freq} onChange={e => setFreq(e.target.value)}>
                  {FREQ_OPTIONS.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Initial Value Type</label>
                <select className="form-select" value={valType} onChange={e => setValType(e.target.value)}>
                  {VALUE_OPTIONS.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Unit</label>
                <select className="form-select" value={valUnit} onChange={e => setValUnit(e.target.value)}>
                  {UNIT_OPTIONS.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div />
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.25rem' }}>
              <button className="btn btn-primary btn-lg" onClick={handleSave}
                disabled={saving || polling || stocks.length === 0} style={{ flex: 1 }}>
                {saving ? '⏳ Saving…' : polling ? `🔄 Training (${pct.toFixed(0)}%)…` : '💾 Save & Retrain Model'}
              </button>
              <button className="btn btn-ghost btn-lg"
                onClick={() => { getPreferences().then(r => {
                  setStocks(r.data.stocks||[]); setFreq(r.data.rebalanceFrequency||'Monthly')
                }) }}>
                Cancel
              </button>
            </div>
          </div>

          {/* Training progress (after save) */}
          {trainJob && (
            <div className="card">
              <div className="card-title" style={{ display:'flex', justifyContent:'space-between' }}>
                <span>Model Retraining Progress</span>
                <span className={`badge badge-${trainJob.status==='COMPLETED'?'done':trainJob.status==='FAILED'?'negative':'running'}`}>
                  {trainJob.status}
                </span>
              </div>
              <div className="progress-wrap">
                <div className="progress-label">
                  <span>{STAGE_LABELS[stage] || stage}</span>
                  <span>{pct.toFixed(1)}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${pct}%` }} />
                </div>
              </div>
              {trainJob.status === 'COMPLETED' && (
                <div className="alert alert-success" style={{ marginTop: '0.5rem' }}>
                  ✅ New model trained and hot-reloaded for your selected stocks.
                </div>
              )}
              {trainJob.status === 'FAILED' && (
                <div className="alert alert-error" style={{ marginTop: '0.5rem' }}>❌ {trainJob.error}</div>
              )}
            </div>
          )}
        </div>

        {/* Constraint check sidebar */}
        <div>
          <div className="constraint-panel">
            <div className="constraint-title">Constraint Check</div>
            <p style={{ fontSize: '0.82rem', marginBottom: '0.75rem',
              color: stocks.length >= 1 && stocks.length <= 10 ? 'var(--green)' : 'var(--red)' }}>
              {stocks.length} constraint{stocks.length !== 1 ? 's' : ''} (stock{stocks.length !== 1 ? 's' : ''}) selected.
            </p>
            <div className="summary-grid">
              <div className="summary-row">
                <span className="summary-key">Min required</span>
                <span className={`summary-value ${stocks.length >= 1 ? 'constraint-ok' : 'constraint-bad'}`}>
                  {stocks.length >= 1 ? '✓' : '✗'} 1
                </span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Max allowed</span>
                <span className={`summary-value ${stocks.length <= 10 ? 'constraint-ok' : 'constraint-bad'}`}>
                  {stocks.length <= 10 ? '✓' : '✗'} 10
                </span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Selected</span>
                <span className="summary-value" style={{ color: constraintColor }}>
                  {stocks.length}
                </span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Remaining slots</span>
                <span className="summary-value">{Math.max(0, 10 - stocks.length)}</span>
              </div>
            </div>
          </div>

          <div className="card" style={{ marginTop: '1rem' }}>
            <div className="card-title">How It Works</div>
            <ol style={{ paddingLeft: '1.2rem', fontSize: '0.82rem',
              color: 'var(--text-secondary)', lineHeight: 2 }}>
              <li>Select 1–10 NSE stocks</li>
              <li>Save configuration</li>
              <li>LSTM trained per new stock</li>
              <li>PPO fine-tuned on your portfolio</li>
              <li>Model hot-reloaded for inference</li>
            </ol>
          </div>

          {stocks.length > 0 && (
            <div className="card" style={{ marginTop: '1rem' }}>
              <div className="card-title">Selected Stocks</div>
              {stocks.map((s, i) => (
                <div key={s} className="summary-row">
                  <span className="summary-key">#{i+1}</span>
                  <span className="summary-value" style={{ color: 'var(--cyan)' }}>{s.replace('.NS','')}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
