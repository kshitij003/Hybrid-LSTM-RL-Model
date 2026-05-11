import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import { checkHealth } from './services/api'
import { AuthProvider, useAuth } from './context/AuthContext'
import Dashboard   from './pages/Dashboard'
import Training    from './pages/Training'
import Backtesting from './pages/Backtesting'
import Predictions from './pages/Predictions'
import Portfolios  from './pages/Portfolios'
import News        from './pages/News'
import { LoginPage, SignupPage } from './pages/Auth'

function Navbar({ connected }) {
  const { user, logout, isAuthenticated } = useAuth()

  return (
    <nav className="navbar">
      <div className="nav-brand">
        <span className="brand-icon">⚡</span>
        <span className="brand-text">Hybrid LSTM-RL Trading</span>
      </div>

      {isAuthenticated && (
        <div className="nav-links">
          {[
            { to: '/',            label: 'Dashboard'   },
            { to: '/training',    label: 'Training'    },
            { to: '/backtesting', label: 'Backtesting' },
            { to: '/predictions', label: 'Predictions' },
            { to: '/portfolios',  label: 'Portfolios'  },
            { to: '/news',        label: 'News'        },
          ].map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {label}
            </NavLink>
          ))}
        </div>
      )}

      <div className="nav-right" style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
        <div className="nav-status">
          <div className={`status-dot ${connected ? '' : 'offline'}`} />
          <span style={{ color: connected ? 'var(--green)' : 'var(--red)', fontSize: '0.78rem' }}>
            {connected ? 'Connected' : 'Offline'}
          </span>
        </div>
        {isAuthenticated ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--cyan)' }}>{user.username}</span>
            <button onClick={logout} className="btn small outline">Logout</button>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '0.5rem' }}>
             <NavLink to="/login" className="btn small primary">Sign In</NavLink>
          </div>
        )}
      </div>
    </nav>
  )
}

function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth()
  if (loading) return <div className="page center">Loading...</div>
  return isAuthenticated ? children : <Navigate to="/login" />
}

function AppContent() {
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const ping = async () => {
      try { await checkHealth(); setConnected(true) }
      catch { setConnected(false) }
    }
    ping()
    const iv = setInterval(ping, 10_000)
    return () => clearInterval(iv)
  }, [])

  return (
    <div className="app">
      <Navbar connected={connected} />
      <Routes>
        <Route path="/login"  element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        
        <Route path="/"            element={<ProtectedRoute><Dashboard   /></ProtectedRoute>} />
        <Route path="/training"    element={<ProtectedRoute><Training    /></ProtectedRoute>} />
        <Route path="/backtesting" element={<ProtectedRoute><Backtesting /></ProtectedRoute>} />
        <Route path="/predictions" element={<ProtectedRoute><Predictions /></ProtectedRoute>} />
        <Route path="/portfolios"  element={<ProtectedRoute><Portfolios  /></ProtectedRoute>} />
        <Route path="/news"        element={<ProtectedRoute><News        /></ProtectedRoute>} />
      </Routes>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </BrowserRouter>
  )
}
