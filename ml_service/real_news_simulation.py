import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from dotenv import load_dotenv
import logging
import sys
import io

# 1. Load environment variables
load_dotenv()

# 2. Force UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from models.multi_stock_env import MultiStockPortfolioEnv
from models.multi_stock_lstm import MultiStockLSTMPredictor
from data.data_handler import DataHandler
from data.feature_engineer import FeatureEngineer
from api.news import fetch_news

logging.getLogger('yfinance').setLevel(logging.ERROR)

def run_real_news_demo():
    print("\n" + "="*80)
    print(" STARTING REAL-NEWS TRADING SIMULATION (LAST 10 DAYS)")
    print("="*80)

    # 1. Configuration
    STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    INITIAL_BALANCE = 10000
    SEQUENCE_LENGTH = 30
    INPUT_DIM = 9 
    NEWS_LIMIT_DAYS = 15 
    
    # LSTM expects 9 features
    lstm_features = [
        'Open', 'High', 'Low', 'Close', 'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'sentiment'
    ]
    # Environment needs Raw_Close for the wallet math
    expected_cols = lstm_features + ['Raw_Close']

    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)
    
    # 2. Load Models
    print("\n Step 1: Loading AI Models...")
    try:
        active_model_path = os.path.join("models", "saved_models", "multi_stock_train_20260422_233512")
        model = PPO.load(active_model_path)
        lstm_paths = {s: os.path.join("models", "saved_models", f"lstm_{s}.pth") for s in STOCKS}
        lstm_predictor = MultiStockLSTMPredictor(lstm_paths, input_dim=INPUT_DIM, hidden_dim=50)
        print(" Models Loaded.")
    except Exception as e:
        print(f" Load Error: {e}")
        return

    # 3. Fetch REAL News
    print(f"\n Step 2: Fetching Real-Time News (Multi-Engine)...")
    real_sentiment_map = {}
    fallback_sentiment = {}

    for ticker in STOCKS:
        try:
            news = fetch_news(ticker, days=NEWS_LIMIT_DAYS)
            daily_scores = {}
            all_scores = []
            
            if news:
                for art in news:
                    dt = art['publishedAt'][:10]
                    sent = art.get('sentiment', {})
                    score = sent.get('score', 0.0) if isinstance(sent, dict) else 0.0
                    daily_scores.setdefault(dt, []).append(score)
                    all_scores.append(score)
                
                real_sentiment_map[ticker] = {d: np.mean(s) for d, s in daily_scores.items()}
                fallback_sentiment[ticker] = np.mean(all_scores) if all_scores else 0.0
            else:
                real_sentiment_map[ticker] = {}
                fallback_sentiment[ticker] = 0.0
        except Exception as e:
            print(f"    Error processing news for {ticker}: {e}")
            real_sentiment_map[ticker] = {}
            fallback_sentiment[ticker] = 0.0

    # 4. Fetch Market Data
    print("\n Step 3: Preparing Dataframes...")
    stock_dfs = {}
    fe = FeatureEngineer()

    for ticker in STOCKS:
        handler = DataHandler(
            symbols=[ticker], 
            start_date=start_date.strftime('%Y-%m-%d'), 
            end_date=end_date.strftime('%Y-%m-%d')
        )
        df = handler.download_data()
        
        import pandas_ta as ta
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        
        df_feat, _ = fe.get_feature_df(df, sentiment_scores=None)
        
        # Inject Real Sentiment
        df_feat['sentiment'] = 0.0 
        for date_str, score in real_sentiment_map[ticker].items():
            if date_str in df_feat.index.astype(str):
                df_feat.loc[df_feat.index.astype(str) == date_str, 'sentiment'] = score
        
        df_feat.loc[df_feat['sentiment'] == 0.0, 'sentiment'] = fallback_sentiment[ticker]
        
        # Ensure all columns exist and are ordered correctly
        stock_dfs[ticker] = df_feat[expected_cols].fillna(0).dropna()
        print(f"   {ticker}: Done.")

    # 5. Run Simulation Loop
    print("\n" + "="*80)
    print(f"{'DATE':<12} | {'CASH':<10} | {'PORTFOLIO':<10} | {'TOP STOCK & SENTIMENT'}")
    print("-" * 80)

    info = {'cash': INITIAL_BALANCE, 'portfolio_value': INITIAL_BALANCE, 'weights': np.zeros(len(STOCKS)+1)}

    try:
        # Pass the 9 LSTM features to the environment
        env = MultiStockPortfolioEnv(stock_dfs, lstm_features, initial_balance=INITIAL_BALANCE, lstm_predictor=lstm_predictor)
        total_rows = len(stock_dfs[STOCKS[0]])
        env.max_steps = total_rows - 1
        obs, _ = env.reset()
        
        start_simulation_step = max(SEQUENCE_LENGTH, total_rows - 11)
        
        for _ in range(SEQUENCE_LENGTH, start_simulation_step):
            action = np.zeros(len(STOCKS) + 1)
            action[-1] = 1.0 
            obs, _, _, _, _ = env.step(action)

        for step in range(start_simulation_step, env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            current_date_ts = stock_dfs[STOCKS[0]].index[env.step_idx - 1]
            date_str = str(current_date_ts)[:10]
            
            weights = info['weights']
            top_stock_idx = np.argmax(weights[:-1])
            top_stock = STOCKS[top_stock_idx]
            top_weight = weights[top_stock_idx]
            
            current_sentiment = stock_dfs[top_stock].iloc[env.step_idx - 1]['sentiment']

            if current_sentiment > 0.05:
                sentiment_msg = f" BULLISH ({current_sentiment:+.2f})"
            elif current_sentiment < -0.05:
                sentiment_msg = f" BEARISH ({current_sentiment:+.2f})"
            else:
                sentiment_msg = f"⚪ NEUTRAL ({current_sentiment:+.2f})"

            print(f"{date_str:<12} | ₹{info['cash']:<9.0f} | ₹{info['portfolio_value']:<9.0f} | {top_stock}: {top_weight:.0%} {sentiment_msg}")
            
    except Exception as e:
        print(f" Simulation Error: {e}")

    print("-" * 80)
    final_val = info['portfolio_value']
    profit = final_val - INITIAL_BALANCE
    print(f"🏁 SIMULATION COMPLETE")
    print(f" Initial: ₹{INITIAL_BALANCE:,.2f} |  Final: ₹{final_val:,.2f}")
    print(f" Profit:  ₹{profit:,.2f} ({profit/INITIAL_BALANCE:.2%})")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_real_news_demo()