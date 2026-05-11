import { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { useNavigate } from 'react-router-dom'

export function LoginPage() {
  const [form, setForm] = useState({ username: '', password: '' })
  const { login } = useAuth()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    const success = await login(form.username, form.password)
    setLoading(false)
    if (success) navigate('/')
  }

  return (
    <div className="auth-page fade-in">
      <div className="card auth-card">
        <h2 className="gold">Sign In</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>Access your AI trading dashboard</p>
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label>Username</label>
            <input 
              type="text" 
              required 
              value={form.username}
              onChange={e => setForm({...form, username: e.target.value})}
              placeholder="Enter your username"
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input 
              type="password" 
              required 
              value={form.password}
              onChange={e => setForm({...form, password: e.target.value})}
              placeholder="••••••••"
            />
          </div>
          <button type="submit" className="btn primary block" disabled={loading}>
            {loading ? 'Authenticating...' : 'Sign In'}
          </button>
        </form>
        
        <p style={{ marginTop: '1.5rem', textAlign: 'center', fontSize: '0.88rem' }}>
          Don't have an account? <span className="gold pointer" onClick={() => navigate('/signup')}>Sign Up</span>
        </p>
      </div>
    </div>
  )
}

export function SignupPage() {
  const [form, setForm] = useState({ username: '', email: '', password: '' })
  const { signup } = useAuth()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    const success = await signup(form.username, form.email, form.password)
    setLoading(false)
    if (success) navigate('/login')
  }

  return (
    <div className="auth-page fade-in">
      <div className="card auth-card">
        <h2 className="gold">Create Account</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>Join the hybrid trading ecosystem</p>
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label>Username</label>
            <input 
              type="text" 
              required 
              value={form.username}
              onChange={e => setForm({...form, username: e.target.value})}
              placeholder="Pick a unique username"
            />
          </div>
          <div className="form-group">
            <label>Email</label>
            <input 
              type="email" 
              required 
              value={form.email}
              onChange={e => setForm({...form, email: e.target.value})}
              placeholder="you@example.com"
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input 
              type="password" 
              required 
              value={form.password}
              onChange={e => setForm({...form, password: e.target.value})}
              placeholder="Min 8 characters"
            />
          </div>
          <button type="submit" className="btn primary block" disabled={loading}>
            {loading ? 'Creating Account...' : 'Sign Up'}
          </button>
        </form>
        
        <p style={{ marginTop: '1.5rem', textAlign: 'center', fontSize: '0.88rem' }}>
          Already have an account? <span className="gold pointer" onClick={() => navigate('/login')}>Sign In</span>
        </p>
      </div>
    </div>
  )
}
