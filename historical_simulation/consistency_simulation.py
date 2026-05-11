import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import sys
import io

# Load environment variables
load_dotenv()

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import project components
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'ml_service'))
from models.multi_stock_env import MultiStockPortfolioEnv
from models.multi_stock_lstm import MultiStockLSTMPredictor
from data.feature_engineer import FeatureEngineer

def run_consistency_simulation():
    print("\n" + "="*80)
    print("🚀 STARTING 480-DAY CONSISTENCY TRADING SIMULATION")
    print("="*80)

    # 1. Configuration
    STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    INITIAL_BALANCE = 100000 
    SEQUENCE_LENGTH = 30
    INPUT_DIM = 9 
    DATA_PATH = os.path.join(PROJECT_ROOT, 'historical_simulation', 'data', 'consistency_dataset.csv')

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] {DATA_PATH} not found. Run the scraper and preparation script first.")
        return

    # 2. Load Models
    print("\n🧠 Step 1: Loading AI Models...")
    try:
        model_dir = os.path.join(PROJECT_ROOT, "ml_service", "models", "saved_models")
        active_model_txt = os.path.join(model_dir, "active_model.txt")
        
        if os.path.exists(active_model_txt):
            with open(active_model_txt, "r") as f:
                active_model_name = f.read().strip()
            active_model_path = os.path.join(model_dir, active_model_name)
        else:
            raise FileNotFoundError("active_model.txt not found.")

        print(f"   Loading PPO: {os.path.basename(active_model_path)}")
        model = PPO.load(active_model_path)
        
        lstm_paths = {s: os.path.join(model_dir, f"lstm_{s}.pth") for s in STOCKS}
        lstm_predictor = MultiStockLSTMPredictor(lstm_paths, input_dim=INPUT_DIM, hidden_dim=50)
        print("✅ Models Loaded.")
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # 3. Load and Feature Engineer Data
    print("\n📊 Step 2: Preparing Dataset...")
    full_df = pd.read_csv(DATA_PATH)
    fe = FeatureEngineer()
    
    stock_dfs = {}
    lstm_features = [
        'Open', 'High', 'Low', 'Close', 'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'sentiment'
    ]

    for ticker in STOCKS:
        df_ticker = full_df[full_df['ticker'] == ticker].copy()
        df_ticker = df_ticker.sort_values('Date')
        df_ticker = df_ticker.set_index('Date')
        
        df_feat, _ = fe.get_feature_df(df_ticker, sentiment_scores=df_ticker['sentiment'].tolist())
        stock_dfs[ticker] = df_feat
        print(f"   {ticker}: {len(df_feat)} days ready.")

    # 4. Initialize Environment
    env = MultiStockPortfolioEnv(stock_dfs, lstm_features, initial_balance=INITIAL_BALANCE, lstm_predictor=lstm_predictor)
    total_rows = len(stock_dfs[STOCKS[0]])
    env.max_steps = total_rows - 1
    obs, _ = env.reset()

    # 5. Simulation Loop
    print("\n" + "="*80)
    print(f"{'DATE':<12} | {'CASH':<10} | {'PORTFOLIO':<10} | {'TOP STOCK & SENTIMENT'}")
    print("-" * 80)

    history = []
    
    # Main simulation
    for step in range(SEQUENCE_LENGTH, env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        current_date = stock_dfs[STOCKS[0]].index[env.step_idx - 1]
        
        weights = info['weights']
        top_stock_idx = np.argmax(weights[:-1])
        top_stock = STOCKS[top_stock_idx]
        top_weight = weights[top_stock_idx]
        
        current_sentiment = stock_dfs[top_stock].iloc[env.step_idx - 1]['sentiment']
        
        sent_label = "⚪"
        if current_sentiment > 0.6: sent_label = "🟢"
        elif current_sentiment < 0.4: sent_label = "🔴"

        if step % 20 == 0 or step > env.max_steps - 5: # Print every 20 days
            print(f"{current_date[:10]:<12} | ₹{info['cash']:<9.0f} | ₹{info['portfolio_value']:<9.0f} | {top_stock}: {top_weight:.0%} {sent_label}")
        
        history.append({
            'date': current_date,
            'portfolio_value': info['portfolio_value'],
            'cash': info['cash']
        })

    # 6. Final Results & Plotting
    print("-" * 80)
    final_val = info['portfolio_value']
    profit = final_val - INITIAL_BALANCE
    print(f"🏁 SIMULATION COMPLETE")
    print(f"💰 Initial: ₹{INITIAL_BALANCE:,.2f} | 📈 Final: ₹{final_val:,.2f}")
    print(f"✨ Profit:  ₹{profit:,.2f} ({profit/INITIAL_BALANCE:.2%})")
    
    # Plotting
    df_history = pd.DataFrame(history)
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df_history['date']), df_history['portfolio_value'], label='Portfolio Value', color='#3498db', linewidth=2)
    plt.axhline(y=INITIAL_BALANCE, color='red', linestyle='--', label='Initial Balance')
    plt.title(f'Consistency Test Simulation (2023-2024)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (INR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    report_path = 'historical_simulation/consistency_results.png'
    plt.savefig(report_path)
    print(f"\n📊 Results plot saved to {report_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_consistency_simulation()
