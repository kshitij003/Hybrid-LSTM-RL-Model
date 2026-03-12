import React from 'react'
import { Link } from 'react-router-dom'
import './Navbar.css'

function Navbar({ apiStatus }) {
  const getStatusColor = () => {
    switch (apiStatus) {
      case 'connected':
        return '#00cc99'
      case 'disconnected':
        return '#ff3333'
      case 'checking':
        return '#ffaa00'
      default:
        return '#a0a0a0'
    }
  }

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <h1>🤖 Hybrid LSTM-RL Trading</h1>
        </div>
        <ul className="navbar-menu">
          <li><Link to="/">Dashboard</Link></li>
          <li><Link to="/training">Training</Link></li>
          <li><Link to="/backtesting">Backtesting</Link></li>
          <li><Link to="/predictions">Predictions</Link></li>
        </ul>
        <div className="navbar-status">
          <div className="status-indicator" style={{ backgroundColor: getStatusColor() }}></div>
          <span className="status-text">
            {apiStatus === 'connected' ? 'Connected' : 
             apiStatus === 'disconnected' ? 'Disconnected' :
             apiStatus === 'checking' ? 'Checking...' : 'Error'}
          </span>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
