import { useState, useEffect, useRef } from 'react'
import { getPreferences, trainCustom, quickUpdate, startTraining, getTrainingStatus } from '../services/api'
import toast from 'react-hot-toast'

const STAGE_LABELS = {
  INITIALIZING: 'Initializing…',
  DATA_DOWNLOAD: 'Downloading market data…',
  LSTM_TRAINING: 'Training LSTM models…',
  LOADING_LSTM:  'Loading LSTM weights…',
  BUILDING_ENV:  'Building trading environment…',
  PPO_FINETUNING:'Fine-tuning PPO agent…',
  FINISHED:      'Complete!',
}

export default function Training() {
  const [prefs, setPrefs]           = useState({ stocks: [] })
  const [form,  setForm]            = useState({
    startDate:    '2023-01-01',
    endDate:      '2025-01-01',
    lstmEpochs:   10,
    ppoTimesteps: 50000,
    sequenceLen:  30,
    initialBal:   100000,
    mode:         'quick',
  })
  const [job,       setJob]         = useState(null)
  const [polling,   setPolling]     = useState(false)
  const [submitting,setSubmitting]  = useState(false)
  const pollRef = useRef(null)

  useEffect(() => {
    getPreferences().then(r => setPrefs(r.data)).catch(() => {})
    return () => clearInterval(pollRef.current)
  }, [])

  const startPoll = (id) => {
    pollRef.current = setInterval(async () => {
      try {
        const r = await getTrainingStatus(id)
        setJob(r.data)
        if (['COMPLETED','FAILED','CANCELLED'].includes(r.data.status)) {
          clearInterval(pollRef.current)
          setPolling(false)
          if (r.data.status === 'COMPLETED') toast.success('Training complete! Model hot-reloaded.')
          else if (r.data.status === 'FAILED') toast.error(`Training failed: ${r.data.error}`)
        }
      } catch { clearInterval(pollRef.current); setPolling(false) }
    }, 2500)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (prefs.stocks.length === 0) { toast.error('No stocks selected. Configure your portfolio first.'); return }
    setSubmitting(true)
    try {
      const config = {
        startDate:      form.startDate,
        endDate:        form.endDate,
        lstmEpochs:     Number(form.lstmEpochs),
        ppoTimesteps:   Number(form.ppoTimesteps),
        sequenceLength: Number(form.sequenceLen),
        initialBalance: Number(form.initialBal),
      }
      let res
      if (form.mode === 'quick') {
        res = await trainCustom({ stocks: prefs.stocks, config })
      } else {
        res = await startTraining({ stocks: prefs.stocks, ...config })
      }
      const jobData = { trainingId: res.data.trainingId, status: 'QUEUED',
                        progress: { stage: 'INITIALIZING', percentComplete: 0 } }
      setJob(jobData)
      setPolling(true)
      startPoll(res.data.trainingId)
      toast.success(`Training job started (${res.data.mode || form.mode.toUpperCase()})`)
    } catch (err) {
      toast.error(err?.response?.data?.error?.message || 'Failed to start training')
    } finally { setSubmitting(false) }
  }

  const pct   = job?.progress?.percentComplete || 0
  const stage = job?.progress?.stage || 'INITIALIZING'

  return (
    <main className="page fade-in">
      <div className="page-header">
        <h1 className="page-title">Model Training</h1>
        <p className="page-subtitle">Train LSTM feature extractors + fine-tune the PPO agent on your selected stocks</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '1.5rem' }}>
        {/* Form */}
        <div className="card">
          <div className="card-title">Training Configuration</div>

          {prefs.stocks.length > 0 ? (
            <div className="alert alert-info" style={{ marginBottom: '1rem' }}>
              🎯 Will train on: <strong>{prefs.stocks.join(', ')}</strong>
            </div>
          ) : (
            <div className="alert alert-error">
              ⚠️ No stocks configured. Go to Portfolios to select stocks first.
            </div>
          )}

          <form onSubmit={handleSubmit}>
            {/* Mode */}
            <div className="form-group">
              <label className="form-label">Training Mode</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {[{v:'quick',l:'Quick Update (LSTM → PPO fine-tune)'},{v:'full',l:'Full Retrain (hours)'}].map(({v,l})=>(
                  <button key={v} type="button"
                    className={`btn ${form.mode===v ? 'btn-primary' : 'btn-ghost'} btn-sm`}
                    style={{ flex:1 }}
                    onClick={() => setForm(f=>({...f, mode:v}))}>
                    {l}
                  </button>
                ))}
              </div>
            </div>

            <div className="form-row">
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
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">LSTM Epochs</label>
                <input className="form-input" type="number" min={1} max={100} value={form.lstmEpochs}
                  onChange={e => setForm(f=>({...f,lstmEpochs:e.target.value}))} />
              </div>
              <div className="form-group">
                <label className="form-label">PPO Timesteps</label>
                <input className="form-input" type="number" min={1000} value={form.ppoTimesteps}
                  onChange={e => setForm(f=>({...f,ppoTimesteps:e.target.value}))} />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Sequence Length</label>
                <input className="form-input" type="number" min={10} max={120} value={form.sequenceLen}
                  onChange={e => setForm(f=>({...f,sequenceLen:e.target.value}))} />
              </div>
              <div className="form-group">
                <label className="form-label">Initial Balance (₹)</label>
                <input className="form-input" type="number" min={1000} value={form.initialBal}
                  onChange={e => setForm(f=>({...f,initialBal:e.target.value}))} />
              </div>
            </div>

            <button className="btn btn-primary btn-lg" type="submit"
              disabled={submitting || polling || prefs.stocks.length === 0}
              style={{ width: '100%', marginTop: '0.5rem' }}>
              {submitting ? '⏳ Starting…' : polling ? '⏳ Training in progress…' : '🚀 Start Training'}
            </button>
          </form>
        </div>

        {/* Progress panel */}
        <div>
          <div className="card" style={{ marginBottom: '1rem' }}>
            <div className="card-title">Training Progress</div>
            {!job ? (
              <div className="empty-state" style={{ padding: '1.5rem 0.5rem' }}>
                <div className="empty-icon">🤖</div>
                <p>No active job.<br/>Start a training run.</p>
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    {job.trainingId}
                  </span>
                  <span className={`badge badge-${
                    job.status==='COMPLETED'?'done':job.status==='IN_PROGRESS'?'running':
                    job.status==='FAILED'?'negative':'idle'}`}>
                    {job.status}
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

                {job.progress?.currentTimestep && (
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                    PPO step: {job.progress.currentTimestep?.toLocaleString()} / {job.progress.totalTimesteps?.toLocaleString()}
                  </p>
                )}

                {job.results && (
                  <div className="alert alert-success" style={{ marginTop: '0.75rem' }}>
                    ✅ Model saved: <code style={{ fontSize: '0.75rem' }}>{job.results.modelPath}</code>
                  </div>
                )}

                {job.status === 'FAILED' && (
                  <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>
                    ❌ {job.error}
                  </div>
                )}
              </>
            )}
          </div>

          <div className="card">
            <div className="card-title">Mode Guide</div>
            <div className="summary-grid">
              <div className="summary-row">
                <span className="summary-key">Quick Update</span>
                <span className="summary-value" style={{ color: 'var(--green)' }}>~15–30 min</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Full Retrain</span>
                <span className="summary-value" style={{ color: 'var(--yellow)' }}>2–3 hrs</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">New stocks added</span>
                <span className="summary-value">LSTM trained fresh</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Existing stocks</span>
                <span className="summary-value">Weights reused</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
