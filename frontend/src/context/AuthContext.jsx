import { createContext, useContext, useState, useEffect } from 'react'
import api, { setToken } from '../services/api'
import { toast } from 'react-hot-toast'

const AuthContext = createContext()

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const savedUser = localStorage.getItem('trader_user')
    const savedToken = localStorage.getItem('trader_token')
    
    if (savedUser && savedToken) {
      const u = JSON.parse(savedUser)
      setUser(u)
      setToken(savedToken)
    }
    setLoading(false)
  }, [])

  const login = async (username, password) => {
    try {
      const res = await api.post('/api/auth/signin', { username, password })
      const { token, ...userData } = res.data
      
      localStorage.setItem('trader_token', token)
      localStorage.setItem('trader_user', JSON.stringify(userData))
      
      setToken(token)
      setUser(userData)
      toast.success(`Welcome back, ${userData.username}!`)
      return true
    } catch (err) {
      toast.error(err.response?.data?.message || 'Login failed')
      return false
    }
  }

  const signup = async (username, email, password) => {
    try {
      await api.post('/api/auth/signup', { username, email, password })
      toast.success('Registration successful! Please sign in.')
      return true
    } catch (err) {
      toast.error(err.response?.data?.message || 'Signup failed')
      return false
    }
  }

  const logout = () => {
    localStorage.removeItem('trader_token')
    localStorage.removeItem('trader_user')
    setToken(null)
    setUser(null)
    toast.success('Logged out successfully')
  }

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, loading, isAuthenticated: !!user }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
