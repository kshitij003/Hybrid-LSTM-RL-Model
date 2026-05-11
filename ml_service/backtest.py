"""
Simple Backtesting Module
Tests trained models against historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from stable_baselines3 import PPO
from models.multi_stock_env import MultiStockPortfolioEnv
import matplotlib.pyplot as plt


class SimpleBacktester:
    """Backtest PPO model on historical data"""
    
    def __init__(self, model_path: str, stocks: list, initial_balance: float = 10000):
        self.model_path = model_path
        self.stocks = stocks
        self.initial_balance = initial_balance
        self.model = None
        
    def load_model(self):
        """Load trained PPO model"""
        try:
            self.model = PPO.load(self.model_path)
            print(f" Model loaded from: {self.model_path}")
            return True
        except Exception as e:
            print(f" Failed to load model: {e}")
            return False
    
    def fetch_backtest_data(self, start_date: str, end_date: str):
        """
        Fetch historical data for backtesting
        
        Args:
            start_date: Format 'YYYY-MM-DD'
            end_date: Format 'YYYY-MM-DD'
        """
        print(f"\n Fetching data from {start_date} to {end_date}...")
        
        stock_data = {}
        
        for ticker in self.stocks:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                df = df[['Close', 'Volume']].copy()
                df.columns = ['close', 'volume']
                df['sentiment'] = 0.0  # Placeholder - would need actual sentiment
                
                stock_data[ticker] = df
                print(f"   {ticker}: {len(df)} days")
                
            except Exception as e:
                print(f"    Failed to fetch {ticker}: {e}")
        
        return stock_data
    
    def run_backtest(self, stock_data: dict):
        """
        Run backtest on historical data
        
        Returns:
            dict: Performance metrics
        """
        print(f"\n Running backtest...")
        
        # Create environment
        env = MultiStockPortfolioEnv(
            stock_data=stock_data,
            initial_balance=self.initial_balance,
            stocks=self.stocks
        )
        
        # Run simulation
        obs, _ = env.reset()
        done = False
        
        portfolio_values = [self.initial_balance]
        actions_taken = []
        
        step = 0
        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            actions_taken.append(action)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Track portfolio value
            portfolio_values.append(info.get('portfolio_value', self.initial_balance))
            
            step += 1
            if step % 10 == 0:
                print(f"   Step {step}: Portfolio = ${portfolio_values[-1]:.2f}")
        
        # Calculate metrics
        final_value = portfolio_values[-1]
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate buy-and-hold baseline
        baseline_value = self.calculate_buy_and_hold(stock_data)
        baseline_return = ((baseline_value - self.initial_balance) / self.initial_balance) * 100
        
        # Risk metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown
        cumulative = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative) / cumulative
        max_drawdown = np.min(drawdown) * 100
        
        results = {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'baseline_return': baseline_return,
            'alpha': total_return - baseline_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_steps': step,
            'portfolio_values': portfolio_values
        }
        
        return results
    
    def calculate_buy_and_hold(self, stock_data: dict):
        """Calculate buy-and-hold baseline return"""
        allocation_per_stock = self.initial_balance / len(self.stocks)
        final_value = 0
        
        for ticker in self.stocks:
            df = stock_data[ticker]
            initial_price = df['close'].iloc[0]
            final_price = df['close'].iloc[-1]
            
            shares = allocation_per_stock / initial_price
            final_value += shares * final_price
        
        return final_value
    
    def print_results(self, results: dict):
        """Print backtest results"""
        print(f"\n{'='*70}")
        print(f" BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"\n Returns:")
        print(f"   Initial Balance: ${results['initial_balance']:,.2f}")
        print(f"   Final Value: ${results['final_value']:,.2f}")
        print(f"   Total Return: {results['total_return']:.2f}%")
        print(f"\n Comparison:")
        print(f"   Buy & Hold Return: {results['baseline_return']:.2f}%")
        print(f"   Alpha (Excess Return): {results['alpha']:.2f}%")
        print(f"\n Risk Metrics:")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"\n Performance:")
        if results['alpha'] > 0:
            print(f"    BEAT THE MARKET by {results['alpha']:.2f}%!")
        else:
            print(f"    Underperformed by {abs(results['alpha']):.2f}%")
        print(f"\n{'='*70}\n")
    
    def plot_results(self, results: dict):
        """Plot portfolio value over time"""
        plt.figure(figsize=(12, 6))
        
        portfolio_values = results['portfolio_values']
        days = list(range(len(portfolio_values)))
        
        # Calculate baseline
        baseline_values = [self.initial_balance * (1 + results['baseline_return']/100 * (i/len(days))) 
                          for i in range(len(days))]
        
        plt.plot(days, portfolio_values, label='PPO Agent', linewidth=2)
        plt.plot(days, baseline_values, label='Buy & Hold', linestyle='--', linewidth=2)
        
        plt.axhline(y=self.initial_balance, color='gray', linestyle=':', alpha=0.5)
        
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Backtest: PPO Agent vs Buy & Hold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"backtest_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f" Chart saved to: {filename}")
        
        plt.show()


def run_simple_backtest():
    """
    Quick backtest example
    """
    print(f"\n{'='*70}")
    print(f" STARTING BACKTEST")
    print(f"{'='*70}\n")
    
    # Configuration
    MODEL_PATH = "models/saved_models/ppo_multi_stock"  # Update with your model path
    STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    INITIAL_BALANCE = 10000
    
    # Date range for backtest (last 6 months)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # Create backtester
    backtester = SimpleBacktester(MODEL_PATH, STOCKS, INITIAL_BALANCE)
    
    # Load model
    if not backtester.load_model():
        print(" Cannot run backtest without model")
        return
    
    # Fetch data
    stock_data = backtester.fetch_backtest_data(start_date, end_date)
    
    if not stock_data:
        print(" No data fetched")
        return
    
    # Run backtest
    results = backtester.run_backtest(stock_data)
    
    # Display results
    backtester.print_results(results)
    
    # Plot
    backtester.plot_results(results)


if __name__ == "__main__":
    run_simple_backtest()
