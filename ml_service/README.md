# ML Service - LSTM+RL Trading System

Flask-based ML service for multi-stock portfolio management using LSTM and Reinforcement Learning.

## Project Structure

```
ml_service/
├── app.py                      # Flask application entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
│
├── models/                     # ML models
│   ├── lstm_model.py          # LSTM architecture
│   ├── single_stock_env.py    # Single stock trading env
│   └── saved_models/          # Trained model files
│
├── api/                        # REST API endpoints
│   ├── inference.py           # Prediction endpoints
│   ├── training.py            # Training endpoints
│   └── models.py              # Model management
│
├── training/                   # Training scripts
│   ├── train_lstm.py          # LSTM training
│   └── train_ppo.py           # PPO training
│
├── data/                       # Data handling
│   ├── data_handler.py        # Data download/cache
│   ├── feature_engineer.py    # Feature engineering
│   └── cache/                 # Cached data files
│
└── utils/                      # Utility functions
```

## Setup

1. **Install Dependencies**:
```bash
cd ml_service
pip install -r requirements.txt
```

2. **Configure Environment**:
Edit `.env` file with your settings

3. **Run the Server**:
```bash
python app.py
```

Server will start on `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "ml-backend",
  "version": "1.0.0",
  "framework": "Flask"
}
```

## Next Steps

- [ ] Implement inference endpoints
- [ ] Create multi-stock environment
- [ ] Add training pipeline
- [ ] Integrate with Spring Boot backend
