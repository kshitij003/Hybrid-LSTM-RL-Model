import { useState, useEffect } from 'react'
import { getNews } from '../services/api'
import toast from 'react-hot-toast'

// Fallback mock data shown while backend is offline / news endpoint unavailable
const MOCK_NEWS = [
  { headline: "Reliance Industries Q3 profit surges 18% on retail growth",    source: "ET Markets",   sentiment: "POSITIVE" },
  { headline: "RBI holds repo rate at 6.5% amid inflation concerns",           source: "Bloomberg",    sentiment: "NEUTRAL"  },
  { headline: "TCS bags $2B mega-deal from European banking consortium",        source: "CNBC-TV18",   sentiment: "POSITIVE" },
  { headline: "IT sector faces headwinds as US client budgets tighten",         source: "Reuters",      sentiment: "NEGATIVE" },
  { headline: "HDFC Bank NPA ratio improves to decade-low level",               source: "Mint",         sentiment: "POSITIVE" },
  { headline: "Adani Group acquires port infrastructure for ₹8,000 Cr",        source: "Bloomberg",    sentiment: "NEUTRAL"  },
  { headline: "Indian pharma exports hit all-time high in FY25",                source: "Financial Express","sentiment":"POSITIVE"},
  { headline: "Global crude oil prices rise on OPEC supply cut",                source: "Reuters",      sentiment: "NEGATIVE" },
]

function SentimentBadge({ sentiment }) {
  const cls = { POSITIVE: 'badge-positive', NEGATIVE: 'badge-negative', NEUTRAL: 'badge-neutral' }
  return <span className={`badge ${cls[sentiment] || 'badge-neutral'}`}>{sentiment}</span>
}

function OverallBar({ positive, negative, neutral }) {
  const total = positive + negative + neutral || 1
  return (
    <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', gap: 2, margin: '0.5rem 0' }}>
      <div style={{ flex: positive/total, background: 'var(--green)', opacity: 0.8 }} />
      <div style={{ flex: neutral/total, background: 'var(--yellow)', opacity: 0.8 }} />
      <div style={{ flex: negative/total, background: 'var(--red)', opacity: 0.8 }} />
    </div>
  )
}

export default function News() {
  const [news,    setNews]    = useState([])
  const [visible, setVisible] = useState(6)
  const [loading, setLoading] = useState(true)
  const [usingMock, setUsingMock] = useState(false)

  useEffect(() => {
    const load = async () => {
      try {
        const r = await getNews()
        const items = r.data.headlines || []
        if (items && items.length > 0) {
          setNews(items)
          setUsingMock(false)
        } else {
          setNews(MOCK_NEWS); setUsingMock(true)
        }
      } catch {
        setNews(MOCK_NEWS); setUsingMock(true)
      } finally { setLoading(false) }
    }
    load()
    const iv = setInterval(load, 60_000)   // refresh every minute
    return () => clearInterval(iv)
  }, [])

  const shown    = news.slice(0, visible)
  const positive = news.filter(n => n.sentiment === 'POSITIVE').length
  const negative = news.filter(n => n.sentiment === 'NEGATIVE').length
  const neutral  = news.filter(n => n.sentiment === 'NEUTRAL').length
  const dominant = positive > negative && positive > neutral ? 'Positive Trend'
                 : negative > positive && negative > neutral ? 'Negative Trend'
                 : 'Neutral / Mixed'

  const overallColor = positive > negative ? 'var(--green)' : negative > positive ? 'var(--red)' : 'var(--yellow)'

  return (
    <main className="page fade-in">
      <div className="page-header">
        <h1 className="page-title">Market News &amp; Sentiment Analysis Feed</h1>
        <p className="page-subtitle">Live NSE market headlines scored by FinBERT sentiment model</p>
      </div>

      {usingMock && (
        <div className="alert alert-info" style={{ marginBottom: '1rem' }}>
          ℹ️ Showing sample headlines — connect the Flask service and set <code>NEWS_API_KEY</code> for live data.
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '1.5rem' }}>
        {/* Main feed */}
        <div>
          <div className="card">
            {loading ? (
              <div className="spinner" />
            ) : (
              <>
                <div className="table-wrap">
                  <table>
                    <thead><tr>
                      <th>Recent Headlines</th>
                      <th>Source</th>
                      <th>Sentiment</th>
                    </tr></thead>
                    <tbody>
                      {shown.map((n, i) => (
                        <tr key={i} style={{ animation: `fadeInUp 0.3s ease ${i * 0.04}s backwards` }}>
                          <td style={{ maxWidth: 480, lineHeight: 1.5 }}>
                            {n.headline || n.title || n.text || '—'}
                          </td>
                          <td style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', whiteSpace: 'nowrap' }}>
                            {n.source || n.publisher || '—'}
                          </td>
                          <td>
                            <SentimentBadge sentiment={
                              n.sentiment ||
                              (n.sentimentScore > 0.1 ? 'POSITIVE' : n.sentimentScore < -0.1 ? 'NEGATIVE' : 'NEUTRAL')
                            } />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {visible < news.length && (
                  <div style={{ textAlign: 'center', marginTop: '1rem' }}>
                    <button className="btn btn-secondary" onClick={() => setVisible(v => v + 6)}>
                      Show More Headlines
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div>
          <div className="card section">
            <div className="card-title">Overall Sentiment Summary</div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
              24h Market Overview
            </p>
            <p style={{ fontSize: '1rem', fontWeight: 700, color: overallColor, marginBottom: '0.75rem' }}>
              {dominant}
            </p>
            <OverallBar positive={positive} negative={negative} neutral={neutral} />
            <div className="summary-grid" style={{ marginTop: '0.75rem' }}>
              <div className="summary-row">
                <span className="summary-key">🟢 Positive</span>
                <span className="summary-value" style={{ color: 'var(--green)' }}>{positive}</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">🟡 Neutral</span>
                <span className="summary-value" style={{ color: 'var(--yellow)' }}>{neutral}</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">🔴 Negative</span>
                <span className="summary-value" style={{ color: 'var(--red)' }}>{negative}</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Total Headlines</span>
                <span className="summary-value">{news.length}</span>
              </div>
            </div>
          </div>

          <div className="card" style={{ marginTop: '1rem' }}>
            <div className="card-title">FinBERT Model Info</div>
            <div className="summary-grid">
              <div className="summary-row">
                <span className="summary-key">Model</span>
                <span className="summary-value">ProsusAI/finbert</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Classes</span>
                <span className="summary-value">Pos / Neg / Neutral</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Refresh</span>
                <span className="summary-value">Every 60 s</span>
              </div>
              <div className="summary-row">
                <span className="summary-key">Source</span>
                <span className="summary-value">NewsAPI (GNews)</span>
              </div>
            </div>
            <button className="btn btn-secondary btn-sm" style={{ marginTop: '0.75rem', width: '100%' }}
              onClick={() => { setLoading(true); getNews().then(r => {
                setNews(r.data.headlines || MOCK_NEWS); setLoading(false)
              }).catch(() => { setNews(MOCK_NEWS); setLoading(false) }) }}>
              🔄 Refresh Now
            </button>
          </div>
        </div>
      </div>
    </main>
  )
}
