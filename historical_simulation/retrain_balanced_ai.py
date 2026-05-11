import os
import sys
import io
import pandas as pd
import numpy as np
from datetime import datetime

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set up project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'ml_service'))

from models.multi_stock_env import MultiStockPortfolioEnv
from models.multi_stock_lstm import MultiStockLSTMPredictor
from data.feature_engineer import FeatureEngineer
from stable_baselines3 import PPO

def run_retraining():
    print("\n" + "="*80)
    print("🚀 STARTING BALANCED AI RETRAINING (Real News | Patience-aware Reward)")
    print("="*80)

    STOCKS    = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    DATA_PATH = os.path.join(PROJECT_ROOT, 'historical_simulation', 'data', 'simulation_dataset.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, "ml_service", "models", "saved_models")

    # ── 0. Sentiment quality gate ─────────────────────────────────────────────
    # Refuse to train on a dataset that is mostly dummy sentiment (= 0.0)
    print("🔍 Checking sentiment quality in simulation_dataset.csv ...")
    _check_df = pd.read_csv(DATA_PATH)
    if 'sentiment' not in _check_df.columns:
        print("[ERROR] 'sentiment' column missing from dataset. Run prepare_simulation_data.py first.")
        sys.exit(1)

    zero_frac = (_check_df['sentiment'] == 0.0).mean()
    real_frac  = 1.0 - zero_frac
    print(f"   Real sentiment coverage : {real_frac:.1%}  (zero/dummy rows: {zero_frac:.1%})")

    if real_frac < 0.70:
        print("\n[HARD STOP] Real-sentiment coverage is below 70%.")
        print("   The model would train on mostly dummy sentiment values — this is not allowed.")
        print("   Run the following pipeline first:")
        print("     1. python historical_simulation/fetch_real_historical_news.py")
        print("     2. python historical_simulation/prepare_simulation_data.py")
        sys.exit(1)

    print(f"   ✅  Sentiment quality check passed ({real_frac:.1%} real data).")
    del _check_df

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("\n📊 Loading Data...")
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

    # ── 2. Load LSTM Predictor ────────────────────────────────────────────────
    print("🧠 Loading LSTM Predictor...")
    lstm_paths = {s: os.path.join(MODEL_DIR, f"lstm_{s}.pth") for s in STOCKS}
    lstm_predictor = MultiStockLSTMPredictor(lstm_paths, input_dim=9, hidden_dim=50)

    # ── 3. Create Environment (with new reward parameters) ───────────────────
    print("🌍 Initialising Environment (patience_window=3, min_dip_threshold=3%)...")
    env = MultiStockPortfolioEnv(
        stock_dataframes  = stock_dfs,
        feature_columns   = lstm_features,
        initial_balance   = 100000,
        lstm_predictor    = lstm_predictor,
        patience_window   = 3,      # must see 3 consecutive red days before selling rewarded
        min_dip_threshold = 0.03,   # 3% drawdown from peak before cash bonus kicks in
    )

    # ── 4. Train PPO ──────────────────────────────────────────────────────────
    # Tuned hyperparameters:
    #   ent_coef  0.005 → agent commits to decisions more cleanly (less random exploration)
    #   n_steps   2048  → collect more experience per update for stable policy gradient
    #   batch_size 256  → larger mini-batch for more stable gradient estimates
    print("🏋️  Training New PPO Agent (150,000 steps)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose        = 1,
        ent_coef       = 0.005,   # reduced entropy → less random, more decisive
        n_steps        = 2048,
        batch_size     = 256,
        learning_rate  = 3e-4,
    )

    model.learn(total_timesteps=150_000)

    # ── 5. Save Model ─────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"multi_stock_balanced_{timestamp}"
    save_path  = os.path.join(MODEL_DIR, model_name)

    print(f"\n💾 Saving model to {save_path}.zip ...")
    model.save(save_path)

    # Update active_model.txt
    active_model_file = os.path.join(MODEL_DIR, "active_model.txt")
    with open(active_model_file, "w") as f:
        f.write(model_name)

    print(f"✅  active_model.txt updated → {model_name}")
    print("="*80)


if __name__ == "__main__":
    run_retraining()
