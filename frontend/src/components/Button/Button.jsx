import React from 'react'
import './Button.css'

function Button({ children, onClick, variant = 'primary', disabled = false, loading = false }) {
  return (
    <button 
      className={`btn btn-${variant}`}
      onClick={onClick}
      disabled={disabled || loading}
    >
      {loading ? '⏳ Loading...' : children}
    </button>
  )
}

export default Button
