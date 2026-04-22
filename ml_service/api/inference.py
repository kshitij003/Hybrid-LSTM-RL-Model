"""
Inference API Blueprint
Handles model predictions for portfolio rebalancing
"""

from flask import Blueprint, request, jsonify
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import os

# Create blueprint
inference_bp = Blueprint('inference', __name__)

# ── Constants matching the training environment ──────────────────────────────
LSTM_HIDDEN_DIM  = 50
LSTM_NUM_LAYERS  = 2
LSTM_SEQ_LEN     = 30   # sequence_length used during training
MODEL_DIR        = "models/saved_models"
MAX_STOCKS       = 5    # Matches the number of stocks used during training

# Default Indian Nifty 50 stocks (configurable via DEFAULT_STOCKS env var).
# Frontend/training pipeline can always pass a custom list — this is just
# the startup default used to pre-load LSTM .pth files.
_DEFAULT_STOCKS_ENV = os.getenv("DEFAULT_STOCKS", "")
STOCKS: List[str] = (
    [s.strip() for s in _DEFAULT_STOCKS_ENV.split(",") if s.strip()]
    if _DEFAULT_STOCKS_ENV
    else ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
)

# ── Global state ─────────────────────────────────────────────────────────────
active_model         = None
active_model_version = "none"
lstm_predictors: Dict = {}    # {"RELIANCE.NS": LSTMPredictor, ...}
lstm_input_dims: Dict = {}    # {"RELIANCE.NS": 10, ...}  auto-detected from .pth
active_tickers: List  = []    # Currently active tickers for inference


# ─────────────────────────────────────────────────────────────────────────────
#  Startup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_input_dim(pth_path: str) -> int:
    """Read the LSTM input_dim from a saved state_dict without loading the full model."""
    import torch
    sd = torch.load(pth_path, map_location="cpu")
    # weight_ih_l0 shape = (4*hidden_dim, input_dim)
    return int(sd["lstm.weight_ih_l0"].shape[1])


def load_active_model() -> bool:
    """Load the PPO model and all per-stock LSTM models for inference."""
    global active_model, active_model_version, lstm_predictors, lstm_input_dims

    # ── 1. PPO model ─────────────────────────────────────────────────────────
    try:
        from stable_baselines3 import PPO
        from api.models import get_active_model_id
        
        model_id = get_active_model_id() or "ppo_multi_stock"
        ppo_path = os.path.join(MODEL_DIR, model_id)
        
        if os.path.exists(ppo_path + ".zip"):
            active_model         = PPO.load(ppo_path)
            active_model_version = model_id
            print(f"✅ Loaded PPO model: {ppo_path}")
        else:
            print(f"⚠️  No PPO model at {ppo_path}")
            return False
    except Exception as e:
        print(f"❌ Error loading PPO model: {e}")
        return False

    # ── 2. Per-stock LSTM models ─────────────────────────────────────────────
    try:
        from models.lstm_model import LSTMStateModel, LSTMPredictor
        import torch

        lstm_predictors = {}
        lstm_input_dims = {}

        for ticker in STOCKS:
            path = os.path.join(MODEL_DIR, f"lstm_{ticker}.pth")
            if not os.path.exists(path):
                print(f"   ⚠️  LSTM model not found for {ticker}: {path}")
                lstm_predictors[ticker] = None
                continue

            # Auto-detect input_dim stored in the .pth weights
            input_dim = _detect_input_dim(path)
            lstm_input_dims[ticker] = input_dim

            lstm_predictors[ticker] = LSTMPredictor(
                model_path = path,
                input_dim  = input_dim,
                hidden_dim = LSTM_HIDDEN_DIM,
                num_layers = LSTM_NUM_LAYERS,
            )
            print(f"   ✅ Loaded LSTM for {ticker}  (input_dim={input_dim})")

    except Exception as e:
        print(f"⚠️  Could not load LSTM models: {e}. Will fall back to zeros.")
        lstm_predictors = {}
        lstm_input_dims = {}

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Market data helpers (fetch directly from yfinance — no DB needed)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_market_data_from_yfinance(tickers: List[str], seq_len: int = None) -> dict:
    """
    Fetch the last `seq_len` trading days of OHLCV data from yfinance for
    each ticker. Returns a dict keyed by ticker, each value being a list of
    row-dicts ready for the existing feature pipeline.

    Returns:
        {
            "RELIANCE.NS": [
                {"date": "2025-04-01", "close": 1420.5, "volume": 5000000, "sentimentScore": 0.0},
                ...
            ],
            ...
        }
    """
    import yfinance as yf
    from datetime import datetime, timedelta

    if seq_len is None:
        seq_len = LSTM_SEQ_LEN

    # Fetch extra calendar days to guarantee seq_len trading days after
    # weekends and Indian market holidays are filtered out.
    buffer_days = seq_len * 3
    end_dt      = datetime.now()
    start_dt    = end_dt - timedelta(days=buffer_days)

    market_data: dict = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                print(f"   ⚠️  yfinance returned empty data for {ticker}")
                market_data[ticker] = []
                continue

            # Flatten MultiIndex if yfinance returns one
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Keep only the last seq_len trading days
            df = df.tail(seq_len)

            rows = []
            for date, row in df.iterrows():
                rows.append({
                    "date"          : str(date)[:10],
                    "close"         : float(row.get("Close",  0.0)),
                    "volume"        : float(row.get("Volume", 0.0)),
                    "sentimentScore": 0.0,   # filled in by _attach_sentiment
                })

            market_data[ticker] = rows
            print(f"   ✅ yfinance: {len(rows)} days fetched for {ticker}")

        except Exception as e:
            print(f"   ❌ yfinance fetch failed for {ticker}: {e}")
            market_data[ticker] = []

    return market_data


def _attach_sentiment_to_market_data(market_data: dict) -> dict:
    """
    For each ticker in market_data, fetch recent FinBERT sentiment scores
    and attach them to each row's 'sentimentScore' field.

    Falls back gracefully (sentimentScore stays 0.0) if:
      - NEWS_API_KEY is not configured
      - NewsAPI returns no articles
      - FinBERT fails for any reason

    Note: 0.0 here is only used for the last ~30 days during inference.
    The LSTM was trained with a consistent simulated signal, so a neutral
    (0.0) score during inference is the safest honest default when real
    data is unavailable.
    """
    from data.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()

    for ticker, rows in market_data.items():
        if not rows:
            continue

        dates = pd.to_datetime([r['date'] for r in rows])
        idx   = pd.DatetimeIndex(dates)

        sentiment_scores = fe.fetch_and_score_sentiment(
            ticker   = ticker,
            df_index = idx,
            days_back = 14,   # last 2 weeks of news is enough for 30-day window
        )

        if sentiment_scores and len(sentiment_scores) == len(rows):
            for i, row in enumerate(rows):
                row['sentimentScore'] = float(sentiment_scores[i])
        # If None, sentimentScore stays 0.0 — inference already handles this

    return market_data


# ─────────────────────────────────────────────────────────────────────────────
#  Predict endpoint
# ─────────────────────────────────────────────────────────────────────────────

@inference_bp.route('/predict', methods=['POST'])
def predict():
    """
    Simplified portfolio rebalancing prediction.

    Spring Boot only needs to send portfolio state + which tickers to use.
    This endpoint fetches market data from yfinance and sentiment from
    NewsAPI/FinBERT internally — no market data storage in Spring Boot required.

    Request Body:
    {
        "currentCash": 100000.0,
        "currentHoldings": {
            "RELIANCE.NS": 25000.0,
            "TCS.NS": 18000.0
        },
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    }

    Response:
    {
        "modelVersion": "v1.0.0",
        "targetWeights": {
            "RELIANCE.NS": 0.25,
            "TCS.NS": 0.20,
            "HDFCBANK.NS": 0.18,
            "INFY.NS": 0.22,
            "ICICIBANK.NS": 0.10,
            "CASH": 0.05
        },
        "confidenceScore": 0.82,
        "metadata": { "timestamp": "...", "numStocks": 5, "currentValue": 143000.0 }
    }
    """
    try:
        if not request.json:
            return jsonify({"error": {"code": "INVALID_REQUEST",
                                      "message": "Request body must be JSON"}}), 400

        data    = request.json
        missing = [f for f in ['currentCash', 'currentHoldings', 'tickers'] if f not in data]
        if missing:
            return jsonify({"error": {"code": "MISSING_FIELDS",
                                      "message": f"Missing: {', '.join(missing)}"}}), 400

        tickers = data.get('tickers', [])
        if not tickers:
            return jsonify({"error": {"code": "INVALID_REQUEST",
                                      "message": "'tickers' list cannot be empty"}}), 400

        if active_model is None:
            return jsonify({"error": {"code": "MODEL_NOT_LOADED",
                                      "message": "No active model loaded. Train one first."}}), 503

        # ── Step 1: Fetch last LSTM_SEQ_LEN trading days from yfinance ─────────
        print(f"📊 Fetching {LSTM_SEQ_LEN} days of market data for: {tickers}")
        market_data = _fetch_market_data_from_yfinance(tickers, seq_len=LSTM_SEQ_LEN)

        # Check that at least some data came back
        empty_tickers = [t for t, rows in market_data.items() if not rows]
        if len(empty_tickers) == len(tickers):
            return jsonify({"error": {"code": "DATA_FETCH_FAILED",
                                      "message": "yfinance returned no data for any ticker. "
                                                 "Check ticker symbols and internet connectivity."}}), 502
        if empty_tickers:
            print(f"   ⚠️  No data for: {empty_tickers} — they will be excluded from observation.")
            for t in empty_tickers:
                del market_data[t]

        # ── Step 2: Attach FinBERT sentiment to each day ───────────────────
        print("📰 Fetching news sentiment via FinBERT...")
        market_data = _attach_sentiment_to_market_data(market_data)

        # ── Step 3: Build observation and run PPO inference ────────────────
        internal_data = {
            'currentCash'    : float(data['currentCash']),
            'currentHoldings': data['currentHoldings'],
            'marketData'     : market_data,
        }

        observation    = prepare_observation(internal_data)
        action, _      = active_model.predict(observation, deterministic=True)
        
        # Only take the relevant parts of the action vector
        # (the indices matching our active tickers + the final cash slot)
        trained_order = ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"]
        stock_tickers  = sorted(market_data.keys())
        
        active_indices = [i for i, t in enumerate(trained_order) if t in market_data]
        
        # Action vector is [stock0, stock1, stock2, stock3, stock4, cash]
        relevant_actions = np.concatenate([
            action[active_indices],        # The stocks we actually provided
            [action[MAX_STOCKS]]           # The cash slot (always at the end)
        ])
        
        weights = normalize_weights(relevant_actions)
        
        target_weights = {t: float(weights[i]) for i, t in enumerate(stock_tickers)}
        target_weights['CASH'] = float(weights[-1])

        confidence     = calculate_confidence(weights, observation)
        total_value    = float(data['currentCash']) + sum(
            float(v) for v in data['currentHoldings'].values()
        )

        return jsonify({
            "modelVersion"   : active_model_version,
            "targetWeights"  : target_weights,
            "confidenceScore": float(confidence),
            "metadata": {
                "timestamp"   : pd.Timestamp.now().isoformat(),
                "numStocks"   : len(stock_tickers),
                "currentValue": total_value,
                "dataSource"  : "yfinance",
                "tickers"     : stock_tickers,
            }
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": {"code": "INTERNAL_ERROR", "message": str(e)}}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering (mirrors FeatureEngineer used during training)
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_dataframe(ticker_data: list, target_input_dim: int) -> pd.DataFrame:
    """
    Reconstruct a feature DataFrame from the raw market data list, replicating
    the FeatureEngineer pipeline used during training.

    Produced features (order matters — must match training):
      Close, High, Low, Open, RSI_14, MACD_12_26_9,
      MACDh_12_26_9, MACDs_12_26_9, sentiment  [+Adj_Close if model has 10 dims]

    We let pandas_ta compute the same indicators so the feature vectors
    are identical in shape and meaning to those seen during training.
    """
    closes    = [d.get('close',  100.0) for d in ticker_data]
    volumes   = [d.get('volume', 1e6)   for d in ticker_data]
    sentiments= [d.get('sentimentScore', 0.0) for d in ticker_data]

    df = pd.DataFrame({
        'Close' : closes,
        'High'  : [c * 1.005 for c in closes],   # approximate OHLC from close
        'Low'   : [c * 0.995 for c in closes],
        'Open'  : closes,
        'Volume': volumes,
    })

    # Technical indicators (same as FeatureEngineer.add_technical_indicators)
    try:
        import pandas_ta as ta
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
    except Exception:
        df['RSI_14']        = 50.0
        df['MACD_12_26_9']  = 0.0
        df['MACDh_12_26_9'] = 0.0
        df['MACDs_12_26_9'] = 0.0

    # Sentiment (fixed seed, simulated — same as training add_sentiment_data)
    np.random.seed(42)
    rw = np.random.randn(len(df)).cumsum()
    sim_sentiment = np.sin(rw / 50) + np.random.normal(0, 0.1, len(df))
    df['sentiment'] = sim_sentiment
    # Override with real sentimentScore if provided
    df['sentiment'] = sentiments

    # Fill NaN / Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    # Select the numeric, non-Volume feature columns (same as FeatureEngineer.get_feature_df)
    feature_cols = [c for c in df.columns
                    if df[c].dtype != object and c != 'Volume']

    # MinMax-normalise each column (same as FeatureEngineer.normalize_data)
    for col in feature_cols:
        mn, mx = df[col].min(), df[col].max()
        if mx - mn > 0:
            df[col] = (df[col] - mn) / (mx - mn)
        else:
            df[col] = 0.5

    df_feat = df[feature_cols].dropna()

    # If the actual feature count doesn't match the stored input_dim, truncate/pad
    actual_dim = len(feature_cols)
    if actual_dim > target_input_dim:
        feature_cols = feature_cols[:target_input_dim]
        df_feat = df_feat[feature_cols]
    elif actual_dim < target_input_dim:
        for i in range(target_input_dim - actual_dim):
            df_feat[f'_pad_{i}'] = 0.0

    return df_feat


def _compute_lstm_state(ticker: str, ticker_data: list) -> np.ndarray:
    """
    Run the per-stock LSTM on the last LSTM_SEQ_LEN rows and return the 50-dim state.
    Falls back to zeros if the model is unavailable.
    """
    predictor = lstm_predictors.get(ticker)
    if predictor is None:
        return np.zeros(LSTM_HIDDEN_DIM, dtype=np.float32)

    input_dim = lstm_input_dims.get(ticker, LSTM_HIDDEN_DIM)

    try:
        df_feat = _build_feature_dataframe(ticker_data, target_input_dim=input_dim)
        matrix  = df_feat.values.astype(np.float32)

        T = matrix.shape[0]
        if T < LSTM_SEQ_LEN:
            pad    = np.zeros((LSTM_SEQ_LEN - T, input_dim), dtype=np.float32)
            matrix = np.vstack([pad, matrix])
        else:
            matrix = matrix[-LSTM_SEQ_LEN:]

        return predictor.get_latent_state(matrix)

    except Exception as e:
        print(f"⚠️  LSTM inference failed for {ticker}: {e}")
        return np.zeros(LSTM_HIDDEN_DIM, dtype=np.float32)


def _compute_recent_returns(close_prices: np.ndarray, n: int = 5) -> np.ndarray:
    """Last n period-over-period returns from a close price array."""
    if len(close_prices) < 2:
        return np.zeros(n, dtype=np.float32)
    rets = (close_prices[1:] - close_prices[:-1]) / (close_prices[:-1] + 1e-8)
    if len(rets) >= n:
        return rets[-n:].astype(np.float32)
    pad = np.zeros(n - len(rets), dtype=np.float32)
    return np.concatenate([pad, rets]).astype(np.float32)


def _compute_risk_metrics(close_prices: np.ndarray) -> np.ndarray:
    """Return [annualised_sharpe, max_drawdown] from a price series."""
    if len(close_prices) < 3:
        return np.array([0.0, 0.0], dtype=np.float32)
    rets   = (close_prices[1:] - close_prices[:-1]) / (close_prices[:-1] + 1e-8)
    std    = float(np.std(rets)) + 1e-8
    sharpe = float(np.clip(np.mean(rets) / std * np.sqrt(252), -5.0, 5.0))
    peak   = np.maximum.accumulate(close_prices)
    dd     = (peak - close_prices) / (peak + 1e-8)
    max_dd = float(np.clip(np.max(dd), 0.0, 1.0))
    return np.array([sharpe, max_dd], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Observation construction
# ─────────────────────────────────────────────────────────────────────────────

def prepare_observation(request_data: dict) -> np.ndarray:
    """
    Build the PPO observation vector from the API request with padding for MAX_STOCKS.

    Layout (Fixed for MAX_STOCKS=10):
      lstm_states(50*10) | norm_prices(10) | portfolio_weights(11) |
      portfolio_state(3) | recent_returns(5) | risk_metrics(2)
    """
    current_cash     = float(request_data['currentCash'])
    current_holdings = request_data['currentHoldings']
    market_data      = request_data['marketData']

    input_tickers = sorted(market_data.keys())
    num_input_stocks = len(input_tickers)
    
    total_value   = current_cash + sum(current_holdings.values())
    initial_balance = 10000.0   # Matches training baseline

    # ── 1. LSTM latent states (Padded to MAX_STOCKS) ─────────────────────────
    # We must align tickers with their trained slots:
    # 0: HDFCBANK, 1: ICICIBANK, 2: INFY, 3: RELIANCE, 4: TCS
    trained_order = ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"]
    
    lstm_parts = []
    for i in range(MAX_STOCKS):
        ticker = trained_order[i]
        if ticker in market_data:
            state = _compute_lstm_state(ticker, market_data[ticker])
            lstm_parts.append(state)
        else:
            # Padded slot for stocks not in the current request
            lstm_parts.append(np.zeros(LSTM_HIDDEN_DIM, dtype=np.float32))
    lstm_states = np.concatenate(lstm_parts).astype(np.float32)

    # ── 2. Normalised current prices (Aligned with trained_order) ────────────
    normalized_prices_list = []
    for i in range(MAX_STOCKS):
        ticker = trained_order[i]
        if ticker in market_data:
            rows = market_data[ticker]
            fp = float(rows[0].get('close', 100.0))
            cp = float(rows[-1].get('close', 100.0))
            normalized_prices_list.append(cp / (fp + 1e-8))
        else:
            normalized_prices_list.append(0.0)
    normalized_prices = np.array(normalized_prices_list, dtype=np.float32)

    # ── 3. Portfolio weights (Aligned with trained_order + 1 cash) ───────────
    weights_list = []
    for i in range(MAX_STOCKS):
        ticker = trained_order[i]
        if ticker in market_data:
            v = float(current_holdings.get(ticker, 0.0))
            weights_list.append(v / total_value if total_value > 0 else 0.0)
        else:
            weights_list.append(0.0)
    weights_list.append(current_cash / total_value if total_value > 0 else 1.0)
    portfolio_weights = np.array(weights_list, dtype=np.float32)

    # ── 4. Portfolio state (Fixed 3) ──────────────────────────────────────────
    portfolio_state = np.array([
        total_value / initial_balance,
        current_cash / initial_balance,
        (total_value - initial_balance) / initial_balance,
    ], dtype=np.float32)

    # ── 5. Recent returns (Fixed 5) ───────────────────────────────────────────
    if num_input_stocks > 0:
        first_ticker = input_tickers[0]
        close_arr    = np.array([d.get('close', 100.0) for d in market_data[first_ticker]], dtype=np.float32)
        recent_returns = _compute_recent_returns(close_arr, n=5)
        risk_metrics = _compute_risk_metrics(close_arr)
    else:
        recent_returns = np.zeros(5, dtype=np.float32)
        risk_metrics = np.zeros(2, dtype=np.float32)

    # ── Concatenate & clip ───────────────────────────────────────────────────
    obs = np.concatenate([
        lstm_states,
        normalized_prices,
        portfolio_weights,
        portfolio_state,
        recent_returns,
        risk_metrics,
    ])
    obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    obs = np.clip(obs, -10.0, 10.0)
    return obs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_weights(action: np.ndarray) -> np.ndarray:
    action     = np.clip(action, 0.0, 1.0)
    action_sum = np.sum(action)
    return action / action_sum if action_sum > 0 else np.ones_like(action) / len(action)


def calculate_confidence(weights: np.ndarray, observation: np.ndarray) -> float:
    entropy     = -np.sum(weights * np.log(weights + 1e-8))
    max_entropy = np.log(len(weights))
    confidence  = 1.0 - (entropy / max_entropy)
    return float(np.clip(0.5 + confidence * 0.45, 0.5, 0.95))


# ─────────────────────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────────────────────

@inference_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status"      : "healthy",
        "modelLoaded" : active_model is not None,
        "modelVersion": active_model_version,
        "lstmLoaded"  : {t: (lstm_predictors.get(t) is not None) for t in STOCKS},
        "lstmInputDims": lstm_input_dims,
    }), 200
