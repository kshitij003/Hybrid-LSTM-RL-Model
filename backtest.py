"""
backtest.py
Runs a backtest using the trained PPO model
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from data_handler import DataHandler
from feature_engineer import FeatureEngineer
from trading_env import StockTradingEnv


# -------------------------
# Load PPO model
# -------------------------
def load_ppo_model(model_path):
    print(f"Loading PPO model: {model_path}")
    model = PPO.load(model_path)
    return model


# -------------------------
# Run Backtest
# -------------------------
def run_backtest(csv_path, ppo_model_path, initial_balance=10000):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Ensure datetime index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.sort_index()

    # Feature engineering
    fe = FeatureEngineer()
    features_df, feature_cols = fe.get_feature_df(df)

    if features_df is None or features_df.empty:
        raise RuntimeError("FeatureEngineer returned empty features")

    features_df = features_df.reset_index(drop=True)

    # Create trading environment
    env = StockTradingEnv(
        df=features_df,
        feature_cols=feature_cols,
        initial_balance=initial_balance,
    )

    # Load PPO model
    model = load_ppo_model(ppo_model_path)

    obs, _ = env.reset()

    terminated = False
    truncated = False
    step = 0

    balance_history = []
    value_history = []
    action_history = []

    print("Starting backtest...")

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # ----- Safe info access -----
        balance = info.get("balance", env.balance)
        shares = info.get("shares", env.shares)

        # price for portfolio calc
        price = features_df["Close"].iloc[min(step, len(features_df) - 1)]

        # Portfolio Value
        portfolio_value = balance + shares * price

        balance_history.append(balance)
        value_history.append(portfolio_value)
        action_history.append(int(action))

        step += 1

    final_value = value_history[-1]
    profit = final_value - initial_balance

    print("\n===== BACKTEST RESULTS =====")
    print(f"Initial Balance : {initial_balance}")
    print(f"Final Portfolio : {final_value:.2f}")
    print(f"Net Profit      : {profit:.2f}")
    print(f"Total Steps     : {step}")
    print("============================\n")

    return balance_history, value_history, action_history
    


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":    

    CSV_PATH = "cached_data.csv"
    PPO_MODEL_PATH = "models/ppo_hybrid_trader.zip"

    run_backtest(
        csv_path=CSV_PATH,
        ppo_model_path=PPO_MODEL_PATH,
        initial_balance=1000000,
    )
