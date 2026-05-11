"""
API-Based 500-Day Evaluation Test
Tests the ML system through the Flask API endpoint
"""

import requests
import time
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://localhost:8000"
STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
INITIAL_BALANCE = 10000
SIMULATION_DAYS = 500


def test_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_portfolio_recommendation(current_cash, current_holdings, stock_data, current_date):
    """Get portfolio recommendation from API"""
    
    # Prepare market data for the last 60 days leading up to current_date
    market_data = {}
    for ticker in STOCKS:
        if ticker in stock_data and current_date in stock_data[ticker].index:
            # Get last 60 days of data
            idx = stock_data[ticker].index.get_loc(current_date)
            start_idx = max(0, idx - 60)
            
            df_slice = stock_data[ticker].iloc[start_idx:idx+1]
            
            # Convert to list of dicts
            data_list = []
            for date, row in df_slice.iterrows():
                data_list.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']),
                    "sentimentScore": np.random.randn() * 0.5  # Random sentiment
                })
            
            market_data[ticker] = data_list
    
    payload = {
        "currentCash": current_cash,
        "currentHoldings": current_holdings,
        "marketData": market_data
    }
    
    try:
        response = requests.post(f"{API_URL}/api/predict", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def fetch_market_data():
    """Fetch historical market data"""
    print(" Fetching 500 days of market data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=700)
    
    stock_data = {}
    for ticker in STOCKS:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            stock_data[ticker] = df
            print(f"    {ticker}: {len(df)} days")
        except Exception as e:
            print(f"    {ticker}: Failed")
            return None
    
    return stock_data


def simulate_trading(stock_data):
    """Simulate 500-day trading using API recommendations"""
    print(f"\n{'='*75}")
    print(f" RUNNING 500-DAY API-BASED SIMULATION")
    print(f"{'='*75}\n")
    print(f" Starting Capital: ${INITIAL_BALANCE:,.2f}")
    print(f" Stocks: {', '.join(STOCKS)}\n")
    
    # Get trading days (intersection of all stocks)
    trading_days = stock_data[STOCKS[0]].index
    for ticker in STOCKS[1:]:
        trading_days = trading_days.intersection(stock_data[ticker].index)
    
    trading_days = sorted(trading_days)[-SIMULATION_DAYS:]
    
    portfolio_values = [INITIAL_BALANCE]
    daily_returns = []
    holdings = {stock: 0 for stock in STOCKS}
    cash = INITIAL_BALANCE
    
    print("Day  | Date       | Portfolio Value | Daily Return | Action")
    print("-" * 80)
    
    for day_idx, current_date in enumerate(trading_days):
        # Calculate current prices
        current_prices = {}
        for ticker in STOCKS:
            if ticker in stock_data and current_date in stock_data[ticker].index:
                current_prices[ticker] = float(stock_data[ticker].loc[current_date, 'Close'])
        
        # Calculate current holdings value (in shares, not dollars)
        holdings_value_dict = {ticker: shares for ticker, shares in holdings.items()}
        
        # Get API recommendation
        recommendation = get_portfolio_recommendation(cash, holdings_value_dict, stock_data, current_date)
        
        if not recommendation or 'error' in recommendation:
            print(f"\n  API call failed at day {day_idx+1}")
            break
        
        # Calculate current portfolio value
        portfolio_value = cash
        for ticker, shares in holdings.items():
            if ticker in current_prices:
                portfolio_value += shares * current_prices[ticker]
        
        # Extract target weights from API response
        target_weights = recommendation.get('targetWeights', {})
        
        # Rebalance based on target weights
        new_holdings = {}
        total_spent = 0
        
        # First, liquidate all holdings
        proceeds = cash
        for ticker, shares in holdings.items():
            if ticker in current_prices and shares > 0:
                proceeds += shares * current_prices[ticker]
        
        # Then buy according to target weights (excluding CASH weight)
        for ticker in STOCKS:
            weight = target_weights.get(ticker, target_weights.get(ticker.upper(), 0))
            target_value = portfolio_value * weight
            
            if ticker in current_prices and current_prices[ticker] > 0:
                shares = int(target_value / current_prices[ticker])
                new_holdings[ticker] = shares
                total_spent += shares * current_prices[ticker]
        
        holdings = new_holdings
        cash = proceeds - total_spent
        
        # Recalculate portfolio value after rebalancing
        portfolio_value = cash
        for ticker, shares in holdings.items():
            if ticker in current_prices:
                portfolio_value += shares * current_prices[ticker]
        
        # Track metrics
        portfolio_values.append(portfolio_value)
        
        if len(portfolio_values) > 1:
            daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2] * 100
            daily_returns.append(daily_return)
        else:
            daily_return = 0
        
        # Print every 50 days
        if (day_idx + 1) % 50 == 0 or day_idx == 0:
            date_str = current_date.strftime('%Y-%m-%d')
            action = "REBALANCE" if any(w > 0 for w in target_weights.values()) else "HOLD"
            print(f"{day_idx+1:3d}  | {date_str} | ${portfolio_value:14,.2f} | {daily_return:+11.3f}% | {action}")
        
        # Small delay to avoid overwhelming API
        if day_idx % 10 == 0:
            time.sleep(0.1)
    
    return portfolio_values, daily_returns


def analyze_results(portfolio_values, daily_returns):
    """Analyze and display results"""
    print(f"\n{'='*75}")
    print(f" FINAL RESULTS")
    print(f"{'='*75}\n")
    
    initial = INITIAL_BALANCE
    final = portfolio_values[-1]
    profit = final - initial
    return_pct = (profit / initial) * 100
    
    print(f" Financial Performance:")
    print(f"   Starting Capital:  ${initial:,.2f}")
    print(f"   Final Portfolio:   ${final:,.2f}")
    print(f"   Total Profit/Loss: ${profit:+,.2f}")
    print(f"   Total Return:      {return_pct:+.2f}%\n")
    
    # Risk metrics
    if len(daily_returns) > 0:
        returns = np.array(daily_returns)
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe = (avg_return / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        # Drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - cummax) / cummax * 100
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_days = np.sum(returns > 0)
        win_rate = (win_days / len(returns)) * 100
        
        print(f" Risk & Performance:")
        print(f"   Sharpe Ratio:      {sharpe:.2f}")
        print(f"   Max Drawdown:      {max_drawdown:.2f}%")
        print(f"   Win Rate:          {win_rate:.1f}%")
        print(f"   Avg Daily Return:  {avg_return:.3f}%\n")
        
        # Assessment
        print(f"{'='*75}")
        if return_pct > 10:
            print(" EXCELLENT! System generated strong profits!")
            grade = "A"
        elif return_pct > 5:
            print(" GOOD! System beat typical market returns!")
            grade = "B"
        elif return_pct > 0:
            print("  PROFITABLE but modest returns")
            grade = "C"
        else:
            print(" LOSS - System needs optimization")
            grade = "F"
        print(f"{'='*75}\n")
        
        # Plot
        print(" Generating chart...")
        fig, ax = plt.subplots(figsize=(12, 6))
        days = list(range(len(portfolio_values)))
        
        ax.plot(days, portfolio_values, linewidth=2, color='#2E86AB')
        ax.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(days, INITIAL_BALANCE, portfolio_values,
                         where=(np.array(portfolio_values) >= INITIAL_BALANCE),
                         alpha=0.3, color='green')
        ax.fill_between(days, INITIAL_BALANCE, portfolio_values,
                         where=(np.array(portfolio_values) < INITIAL_BALANCE),
                         alpha=0.3, color='red')
        
        ax.set_title(f'API-Based Trading - {len(portfolio_values)-1} Days (Grade: {grade})', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        
        filename = f"api_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f" Chart saved: {filename}\n")
        
        print(f" SUMMARY: ${initial:,.2f} → ${final:,.2f} ({return_pct:+.2f}%) | Sharpe: {sharpe:.2f}\n")


def main():
    print(f"\n{'='*75}")
    print(f" API-BASED 500-DAY EVALUATION")
    print(f"   Using Flask API /api/predict endpoint")
    print(f"{'='*75}\n")
    
    # Step 1: Check API health
    print(" Checking API status...")
    if not test_api_health():
        print(" API is not running!")
        print("   Please start the Flask app: python app.py")
        return
    print(" API is running\n")
    
    # Step 2: Fetch market data
    stock_data = fetch_market_data()
    if not stock_data:
        print(" Failed to fetch market data")
        return
    
    # Step 3: Simulate trading
    portfolio_values, daily_returns = simulate_trading(stock_data)
    
    # Step 4: Analyze results
    analyze_results(portfolio_values, daily_returns)


if __name__ == "__main__":
    main()
