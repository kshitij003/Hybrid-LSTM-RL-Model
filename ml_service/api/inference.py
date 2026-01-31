"""
Inference API Blueprint
Handles model predictions for portfolio rebalancing
"""

from flask import Blueprint, request, jsonify
from typing import Dict, List
import numpy as np
import pandas as pd
import os

# Create blueprint
inference_bp = Blueprint('inference', __name__)

# Global model instance (loaded at startup)
active_model = None
active_model_version = "none"


def load_active_model():
    """Load the active PPO model for inference"""
    global active_model, active_model_version
    
    try:
        from stable_baselines3 import PPO
        model_path = "models/saved_models/ppo_multi_stock"
        
        if os.path.exists(model_path + ".zip"):
            active_model = PPO.load(model_path)
            active_model_version = "v1.0.0"
            print(f"✅ Loaded PPO model: {model_path}")
            return True
        else:
            print(f"⚠️  No trained model found at {model_path}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@inference_bp.route('/predict', methods=['POST'])
def predict():
    """
    Portfolio rebalancing prediction endpoint
    
    Request Body:
    {
        "currentCash": 5000.0,
        "currentHoldings": {"AAPL": 2000, "MSFT": 2000, "GOOGL": 1000},
        "marketData": {
            "AAPL": [{"date": "2024-01-01", "close": 150, "volume": 1M, "sentimentScore": 0.5}, ...],
            "MSFT": [...],
            "GOOGL": [...]
        }
    }
    
    Response:
    {
        "modelVersion": "v1.0.0",
        "targetWeights": {"AAPL": 0.35, "MSFT": 0.30, "GOOGL": 0.20, "CASH": 0.15},
        "confidenceScore": 0.82,
        "metadata": {...}
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Request body must be JSON"
                }
            }), 400
        
        data = request.json
        
        # Validate required fields
        required_fields = ['currentCash', 'currentHoldings', 'marketData']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                "error": {
                    "code": "MISSING_FIELDS",
                    "message": f"Missing required fields: {', '.join(missing_fields)}"
                }
            }), 400
        
        # Check if model is loaded
        if active_model is None:
            return jsonify({
                "error": {
                    "code": "MODEL_NOT_LOADED",
                    "message": "No active model loaded. Please train or activate a model first."
                }
            }), 503
        
        # Prepare observation from request data
        observation = prepare_observation(data)
        
        # Get prediction from PPO model
        action, _states = active_model.predict(observation, deterministic=True)
        
        # Convert action to portfolio weights
        weights = normalize_weights(action)
        
        # Map weights to stock tickers
        stock_tickers = sorted(data['marketData'].keys())
        target_weights = {}
        
        for i, ticker in enumerate(stock_tickers):
            target_weights[ticker] = float(weights[i])
        
        # Cash weight
        target_weights['CASH'] = float(weights[-1])
        
        # Calculate confidence (simple heuristic for now)
        confidence = calculate_confidence(weights, observation)
        
        # Build response
        response = {
            "modelVersion": active_model_version,
            "targetWeights": target_weights,
            "confidenceScore": float(confidence),
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "numStocks": len(stock_tickers),
                "currentValue": float(data['currentCash']) + sum(data['currentHoldings'].values())
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        import traceback
        print(f"Error in predict endpoint: {e}")
        traceback.print_exc()
        
        return jsonify({
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        }), 500


def prepare_observation(request_data: dict) -> np.ndarray:
    """
    Convert request data to observation format for PPO model
    
    Matches the MultiStockPortfolioEnv observation space:
    - LSTM latent states (50 dims × num_stocks) 
    - Current prices (normalized)
    - Portfolio weights
    - Portfolio state
    - Recent returns
    - Risk metrics
    """
    import pandas as pd
    
    # Extract components
    current_cash = float(request_data['currentCash'])
    current_holdings = request_data['currentHoldings']
    market_data = request_data['marketData']
    
    stock_tickers = sorted(market_data.keys())
    num_stocks = len(stock_tickers)
    
    # Calculate current portfolio value
    total_value = current_cash + sum(current_holdings.values())
    
    # 1. LSTM latent states (50 dims per stock) - using zeros for now
    # TODO: Load LSTM models and compute actual latent states for production
    lstm_states = np.zeros(50 * num_stocks, dtype=np.float32)
    
    # 2. Current prices (normalized by 100)
    current_prices = []
    for ticker in stock_tickers:
        ticker_data = market_data[ticker]
        if len(ticker_data) > 0:
            current_prices.append(ticker_data[-1].get('close', 100.0))
        else:
            current_prices.append(100.0)
    
    normalized_prices = np.array(current_prices, dtype=np.float32) / 100.0
    
    # 3. Current portfolio weights (stocks + cash)
    current_weights = []
    for ticker in stock_tickers:
        holding_value = current_holdings.get(ticker, 0.0)
        weight = holding_value / total_value if total_value > 0 else 0.0
        current_weights.append(weight)
    
    # Cash weight
    cash_weight = current_cash / total_value if total_value > 0 else 1.0
    current_weights.append(cash_weight)
    portfolio_weights = np.array(current_weights, dtype=np.float32)
    
    # 4. Portfolio state (normalized)
    portfolio_state = np.array([
        total_value / 10000.0,      # Normalized total value
        current_cash / 10000.0,      # Normalized cash
        0.0                          # Current return (placeholder)
    ], dtype=np.float32)
    
    # 5. Recent returns (last 5 periods) - using zeros as placeholder
    recent_returns = np.zeros(5, dtype=np.float32)
    
    # 6. Risk metrics (Sharpe ratio, max drawdown) - using zeros as placeholder
    risk_metrics = np.array([0.0, 0.0], dtype=np.float32)
    
    # Concatenate all components
    observation = np.concatenate([
        lstm_states,          # 250 dims (50 × 5 stocks)
        normalized_prices,    # 5 dims
        portfolio_weights,    # 6 dims (5 stocks + cash)
        portfolio_state,      # 3 dims
        recent_returns,       # 5 dims
        risk_metrics          # 2 dims
    ])
    
    # Total: 250 + 5 + 6 + 3 + 5 + 2 = 271 dims (for 5 stocks)
    # Clip and handle NaN
    observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
    observation = np.clip(observation, -10.0, 10.0)
    
    return observation


def normalize_weights(action: np.ndarray) -> np.ndarray:
    """Normalize action to valid weights (sum to 1.0)"""
    action = np.clip(action, 0.0, 1.0)
    action_sum = np.sum(action)
    
    if action_sum > 0:
        return action / action_sum
    else:
        # Default: equal weights
        return np.ones_like(action) / len(action)


def calculate_confidence(weights: np.ndarray, observation: np.ndarray) -> float:
    """
    Calculate confidence score for the prediction
    
    This is a simple heuristic. In production, you might use:
    - Model ensemble variance
    - Historical performance correlation
    - Market regime detection
    """
    # Simple heuristic: higher confidence if weights are decisive (not too uniform)
    entropy = -np.sum(weights * np.log(weights + 1e-8))
    max_entropy = np.log(len(weights))
    
    # Inverse of normalized entropy
    confidence = 1.0 - (entropy / max_entropy)
    
    # Scale to reasonable range [0.5, 0.95]
    confidence = 0.5 + confidence * 0.45
    
    return confidence


@inference_bp.route('/health', methods=['GET'])
def health():
    """Health check for inference service"""
    return jsonify({
        "status": "healthy",
        "modelLoaded": active_model is not None,
        "modelVersion": active_model_version
    }), 200
