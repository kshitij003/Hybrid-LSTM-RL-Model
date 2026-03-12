# Hybrid LSTM-RL Trading System - Installation & Setup Guide

## 📋 Project Structure

```
Hybrid-LSTM-RL-Project/
├── backend/                  # FastAPI Backend
│   ├── main.py              # FastAPI application
│   ├── requirements.txt      # Python dependencies
│   └── app/
│       ├── routes/          # API endpoints
│       │   ├── health.py
│       │   ├── training.py
│       │   ├── backtesting.py
│       │   └── prediction.py
│       ├── models/          # Data models
│       └── utils/           # Utility functions
│
├── frontend/                 # React Frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── Navbar/
│   │   │   ├── Card/
│   │   │   └── Button/
│   │   ├── pages/           # Page components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Training.jsx
│   │   │   ├── Backtesting.jsx
│   │   │   └── Predictions.jsx
│   │   ├── services/        # API client
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
├── train.py                 # Training script
├── backtest.py              # Backtesting script
├── trading_env.py           # Trading environment
├── data_handler.py          # Data handling
├── feature_engineer.py      # Feature engineering
├── lstm_state.py            # LSTM state management
└── requirements.txt         # Python dependencies
```

## 🚀 Getting Started

### Step 1: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Step 3: Run the Backend

```bash
cd backend
python main.py
```

The backend will start at: `http://localhost:8000`

API Documentation available at: `http://localhost:8000/docs`

### Step 4: Run the Frontend (in a new terminal)

```bash
cd frontend
npm run dev
```

The frontend will start at: `http://localhost:3000`

## 📊 Feature Overview

### Dashboard

- Real-time system status
- Training progress monitoring
- Latest backtest results
- Portfolio performance metrics

### Training Module

- Configure LSTM-RL model parameters
- Select stock symbols and date ranges
- Monitor training progress
- Cancel training if needed

### Backtesting Module

- Run historical backtests
- Compare different time periods
- View detailed performance metrics
  - Win Rate
  - Max Drawdown
  - Sharpe Ratio
  - Total Returns
- Backtest history tracking

### Predictions Module

- Single stock price predictions
- Batch predictions for multiple symbols
- Confidence scoring
- Market factor analysis
  - Momentum
  - Volatility
  - Trend

## 🔌 API Endpoints

### Health Check

- `GET /api/health` - API status
- `GET /api/status` - System status

### Training

- `POST /api/train` - Start training
- `GET /api/training-status` - Get training status
- `POST /api/train/cancel` - Cancel training

### Backtesting

- `POST /api/backtest` - Run backtest
- `GET /api/backtest/history` - Get history
- `GET /api/backtest/{id}/details` - Get details

### Predictions

- `POST /api/predict` - Single prediction
- `POST /api/predict/batch` - Batch predictions
- `GET /api/predict/history` - Prediction history

## 🛠️ Development

### Backend Development

- Using **FastAPI** for modern async API
- **Pydantic** for data validation
- **SQLAlchemy** ready for database integration
- **CORS** enabled for frontend communication

### Frontend Development

- **React 18** with functional components
- **Vite** for fast development and build
- **React Router** for navigation
- **Axios** for API communication
- **Recharts** integration ready for charts
- Dark theme with modern UI

## 📦 Dependencies

### Backend

- fastapi==0.104.1
- uvicorn==0.24.0
- torch>=2.0.0 (for LSTM)
- stable-baselines3>=2.2.1 (for RL)
- numpy, pandas, scikit-learn
- yfinance for market data

### Frontend

- react@18.2.0
- react-dom@18.2.0
- react-router-dom@6.18.0
- axios@1.6.0
- recharts@2.10.0 (for charts)

## 🔐 Configuration

### Backend Environment Variables

Create `.env` in backend folder:

```
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### Frontend Configuration

API proxy is configured in `vite.config.js` to forward `/api` requests to `http://localhost:8000`

## 📈 Next Steps

1. **Integrate Real Data**: Connect to live market data feeds
2. **Add Database**: Implement persistent storage
3. **User Authentication**: Add login/signup
4. **Risk Management**: Implement portfolio risk controls
5. **Real-time Updates**: Add WebSocket support
6. **Deployment**: Deploy to cloud (AWS, GCP, Azure)

## 🐛 Troubleshooting

### Backend won't start

```bash
# Check if port 8000 is in use
netstat -an | grep 8000

# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Frontend can't connect to backend

- Check if backend is running on `http://localhost:8000`
- Check CORS headers in browser console
- Verify proxy settings in `vite.config.js`

### Missing dependencies

```bash
# Backend
cd backend && pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

## 📝 License

This project is part of the Hybrid LSTM-RL Model trading system.

## 👨‍💻 Development Notes

- Frontend uses Vite for fast HMR (Hot Module Replacement)
- Backend uses async/await with FastAPI
- Both use modern patterns and best practices
- Ready for scaling and production deployment

---

**Last Updated**: March 2024
**Status**: ✅ Ready for Development & Testing
