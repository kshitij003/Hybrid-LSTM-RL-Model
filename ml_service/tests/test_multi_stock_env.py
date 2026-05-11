"""
Test script for Multi-Stock Portfolio Environment
Tests the environment without trained LSTM models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from models.multi_stock_env import MultiStockPortfolioEnv
from models.multi_stock_lstm import MultiStockLSTMPredictor


def create_mock_stock_data(ticker: str, num_days: int = 200) -> pd.DataFrame:
    """Create mock stock data for testing"""
    np.random.seed(hash(ticker) % 10000)
    
    # Generate random walk prices
    returns = np.random.randn(num_days) * 0.02 + 0.0002  # 2% daily vol, slight upward drift
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    df = pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.randn(num_days) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(num_days)) * 0.01),
        'Low': prices * (1 - np.abs(np.random.randn(num_days)) * 0.01),
        'Volume': np.random.randint(1000000, 10000000, num_days)
    })
    
    # Add features (normalized)
    df['returns'] = df['Close'].pct_change().fillna(0)
    df['volume_norm'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
    df['price_norm'] = (df['Close'] - df['Close'].mean()) / df['Close'].std()
    
    return df


def test_environment_initialization():
    """Test 1: Environment initialization"""
    print("\n" + "="*60)
    print("TEST 1: Environment Initialization")
    print("="*60)
    
    # Create mock data for 3 stocks
    stocks = ["AAPL", "MSFT", "GOOGL"]
    stock_dfs = {ticker: create_mock_stock_data(ticker) for ticker in stocks}
    
    feature_cols = ['Close', 'returns', 'volume_norm', 'price_norm']
    
    # Create environment
    env = MultiStockPortfolioEnv(
        stock_dataframes=stock_dfs,
        feature_columns=feature_cols,
        initial_balance=10000.0
    )
    
    print(f" Environment created successfully")
    print(f"   Stocks: {env.stock_tickers}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    return env


def test_reset_and_observation(env):
    """Test 2: Reset and observation"""
    print("\n" + "="*60)
    print("TEST 2: Reset and Observation")
    print("="*60)
    
    obs, info = env.reset()
    
    print(f" Environment reset successfully")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Expected shape: {env.observation_space.shape}")
    print(f"   Initial portfolio value: ${env.portfolio_value:,.2f}")
    print(f"   Initial cash: ${env.cash:,.2f}")
    
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch!"
    print(f" Observation shape matches expected")
    
    return obs


def test_random_actions(env, num_steps=10):
    """Test 3: Random actions"""
    print("\n" + "="*60)
    print(f"TEST 3: Random Actions ({num_steps} steps)")
    print("="*60)
    
    env.reset()
    
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        
        # Normalize to sum to 1
        action = action / np.sum(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")
            print(f"  Return: {info['return']:+.2%}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Weights: {info['weights']}")
        
        if terminated or truncated:
            print(f"\n Episode ended at step {step}")
            break
    
    print(f"\n Completed {step + 1} steps successfully")
    return info


def test_specific_strategy(env):
    """Test 4: Specific strategy (60/30/10 allocation)"""
    print("\n" + "="*60)
    print("TEST 4: Specific Strategy (60/30/10 + 0% cash)")
    print("="*60)
    
    env.reset()
    
    # Fixed allocation: 60% AAPL, 30% MSFT, 10% GOOGL, 0% cash
    fixed_action = np.array([0.6, 0.3, 0.1, 0.0])
    
    total_return = 0
    
    for step in range(20):
        obs, reward, terminated, truncated, info = env.step(fixed_action)
        total_return = info['return']
        
        if step in [0, 9, 19]:
            print(f"\nStep {step}:")
            print(f"  Value: ${info['portfolio_value']:,.2f}")
            print(f"  Return: {total_return:+.2%}")
            print(f"  Actual weights: {info['weights']}")
        
        if terminated or truncated:
            break
    
    print(f"\n Strategy executed successfully")
    print(f"   Final return: {total_return:+.2%}")


def test_with_lstm_predictor(env):
    """Test 5: With LSTM predictor (dummy)"""
    print("\n" + "="*60)
    print("TEST 5: With Dummy LSTM Predictor")
    print("="*60)
    
    # Create mock data
    stocks = ["AAPL", "MSFT", "GOOGL"]
    stock_dfs = {ticker: create_mock_stock_data(ticker) for ticker in stocks}
    feature_cols = ['Close', 'returns', 'volume_norm', 'price_norm']
    
    # Create dummy LSTM predictor
    lstm_predictor = MultiStockLSTMPredictor.create_dummy_predictor(
        stock_tickers=stocks,
        input_dim=len(feature_cols),
        hidden_dim=50
    )
    
    # Create environment with LSTM
    env_with_lstm = MultiStockPortfolioEnv(
        stock_dataframes=stock_dfs,
        feature_columns=feature_cols,
        initial_balance=10000.0,
        lstm_predictor=lstm_predictor
    )
    
    obs, info = env_with_lstm.reset()
    
    print(f" Environment with LSTM created")
    print(f"   Observation shape: {obs.shape}")
    print(f"   (Includes LSTM latent states)")
    
    # Run a few steps
    for _ in range(5):
        action = env_with_lstm.action_space.sample()
        action = action / np.sum(action)
        obs, reward, terminated, truncated, info = env_with_lstm.step(action)
    
    print(f" Successfully ran with LSTM predictor")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("MULTI-STOCK PORTFOLIO ENVIRONMENT - TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Initialization
        env = test_environment_initialization()
        
        # Test 2: Reset and observation
        obs = test_reset_and_observation(env)
        
        # Test 3: Random actions
        info = test_random_actions(env, num_steps=10)
        
        # Test 4: Specific strategy
        test_specific_strategy(env)
        
        # Test 5: With LSTM
        test_with_lstm_predictor(env)
        
        print("\n" + "="*70)
        print(" ALL TESTS PASSED SUCCESSFULLY!")
        print("="*70)
        print("\nYour multi-stock environment is working correctly! ")
        print("Next: Train LSTM models and PPO agent")
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
