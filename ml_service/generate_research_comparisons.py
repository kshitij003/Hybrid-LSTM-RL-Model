"""
Research Comparison Generator
Generates a professional graph comparing:
1. Hybrid LSTM-RL (Our Model)
2. Moving Average Crossover (SMA) - Represents "Older" technical models
3. Buy & Hold Index - Represents the "Passive" baseline
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import time
from datetime import datetime, timedelta

# Configuration — Indian Nifty 50 stocks (NSE suffix for yfinance)
STOCKS          = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
INITIAL_BALANCE = 100000    # ₹1,00,000 (INR)
SIMULATION_DAYS = 500
API_URL         = "http://localhost:8000/api/predict"

def fetch_stock_data(ticker, start, end):
    """Fetch with retries to handle yfinance timeouts"""
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty and len(df) > 100:
                print(f"   ✅ {ticker} loaded ({len(df)} days)")
                return df
            print(f"   ⚠️  {ticker} data incomplete (Attempt {attempt+1}/3)")
        except Exception as e:
            print(f"   ⚠️  {ticker} error: {str(e)[:50]} (Attempt {attempt+1}/3)")
        time.sleep(1)
    return None

def fetch_data():
    print(f"📊 Fetching data for {STOCKS}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=750) 
    
    data = {}
    for ticker in STOCKS:
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            data[ticker] = df
    return data

def get_safe_price(df, date):
    """Safely extract price from a potentially MultiIndex or Series result"""
    try:
        val = df.loc[date, 'Close']
        if hasattr(val, 'values'):
            # If it's a Series or DataFrame
            v = val.values.flatten()
            return float(v[0]) if len(v) > 0 else 0.0
        return float(val)
    except Exception:
        return 0.0

def simulate_sma_crossover(stock_data, trading_days):
    """Traditional SMA-20/SMA-50 crossover strategy"""
    print("📈 Simulating Traditional SMA Crossover...")
    portfolio_values = []
    cash = INITIAL_BALANCE
    holdings = {ticker: 0 for ticker in STOCKS}
    
    for current_date in trading_days:
        portfolio_val = cash
        for ticker in STOCKS:
            df = stock_data[ticker]
            df_past = df.loc[:current_date]
            if len(df_past) < 50:
                continue
                
            current_price = get_safe_price(df, current_date)
            if current_price == 0: continue
            
            # Safe SMA calculation
            close_series = df_past['Close']
            if hasattr(close_series, 'iloc') and len(close_series.shape) > 1:
                close_series = close_series.iloc[:, 0]
                
            sma20 = close_series.tail(20).mean()
            sma50 = close_series.tail(50).mean()
            
            allocation_per_stock = (INITIAL_BALANCE / len(STOCKS))
            if sma20 > sma50:
                if holdings[ticker] == 0:
                    shares = int(allocation_per_stock / current_price)
                    cash -= shares * current_price
                    holdings[ticker] = shares
            else:
                if holdings[ticker] > 0:
                    cash += holdings[ticker] * current_price
                    holdings[ticker] = 0
            
            portfolio_val += holdings[ticker] * current_price
        portfolio_values.append(portfolio_val)
    
    return portfolio_values

def simulate_buy_and_hold(stock_data, trading_days):
    """Passive Buy & Hold strategy"""
    print("📈 Simulating Passive Buy & Hold Index...")
    allocation = INITIAL_BALANCE / len(STOCKS)
    
    start_date = trading_days[0]
    holdings = {}
    for ticker in STOCKS:
        price = get_safe_price(stock_data[ticker], start_date)
        holdings[ticker] = allocation / (price if price > 0 else 1.0)
        
    portfolio_values = []
    for current_date in trading_days:
        val = 0
        for ticker in STOCKS:
            price = get_safe_price(stock_data[ticker], current_date)
            val += holdings[ticker] * price
        portfolio_values.append(val)
    
    return portfolio_values

def get_hybrid_results(stock_data, trading_days):
    """Fetch Hybrid RL results via API"""
    print("🤖 Fetching Hybrid LSTM-RL decisions (via API)...")
    portfolio_values = []
    cash = INITIAL_BALANCE
    holdings = {ticker: 0 for ticker in STOCKS}
    
    for i, current_date in enumerate(trading_days):
        portfolio_val = cash
        prices = {}
        for ticker in STOCKS:
            p = get_safe_price(stock_data[ticker], current_date)
            prices[ticker] = p
            portfolio_val += holdings[ticker] * p
        
        if i == 0 or i % 5 == 0:
            market_data_payload = {}
            for ticker in STOCKS:
                # Ensure we handle dates that might be missing in a specific ticker's index
                df_ticker = stock_data[ticker]
                df_slice = df_ticker[df_ticker.index <= current_date].tail(60)
                
                # Ensure we have a simple close series
                close_s = df_slice['Close']
                if hasattr(close_s, 'iloc') and len(close_s.shape) > 1:
                    close_s = close_s.iloc[:, 0]
                vol_s = df_slice['Volume']
                if hasattr(vol_s, 'iloc') and len(vol_s.shape) > 1:
                    vol_s = vol_s.iloc[:, 0]
                
                market_data_payload[ticker] = [
                    {"date": d.strftime('%Y-%m-%d'), "close": float(v), "volume": int(vol_s.iloc[i])}
                    for i, (d, v) in enumerate(close_s.items())
                ]
            
            payload = {
                "currentCash": cash,
                "currentHoldings": {t: h * prices[t] for t, h in holdings.items()},
                "marketData": market_data_payload
            }
            
            try:
                res = requests.post(API_URL, json=payload, timeout=15).json()
                weights = res.get('targetWeights', {})
                if weights:
                    new_holdings = {}
                    total_spent = 0
                    for ticker in STOCKS:
                        w = weights.get(ticker, 0)
                        target_v = portfolio_val * w
                        if prices[ticker] > 0:
                            shares = int(target_v / prices[ticker])
                            new_holdings[ticker] = shares
                            total_spent += shares * prices[ticker]
                        else:
                            new_holdings[ticker] = 0
                    
                    holdings = new_holdings
                    cash = portfolio_val - total_spent
            except Exception:
                pass 
                
        portfolio_values.append(portfolio_val)
        if i % 100 == 0: print(f"   Progress: {i}/{len(trading_days)}")
        
    return portfolio_values

def main():
    stock_data = fetch_data()
    if not stock_data: return
    
    common_days = None
    for ticker in stock_data:
        if common_days is None:
            common_days = stock_data[ticker].index
        else:
            common_days = common_days.intersection(stock_data[ticker].index)
    
    if common_days is None or len(common_days) < SIMULATION_DAYS:
        print("❌ Not enough common trading days.")
        return
        
    trading_days = sorted(common_days)[-SIMULATION_DAYS:]
    
    hybrid_values = get_hybrid_results(stock_data, trading_days)
    bh_values = simulate_buy_and_hold(stock_data, trading_days)
    sma_values = simulate_sma_crossover(stock_data, trading_days)
    
    print("\n📊 Generating Research Comparison Plot...")
    plt.figure(figsize=(12, 7))
    
    x = range(len(trading_days))
    plt.plot(x, hybrid_values, label='Hybrid LSTM-RL (Ours)', linewidth=3, color='#E63946')
    plt.plot(x, sma_values, label='SMA Crossover (Older Model)', linewidth=2, color='#457B9D', linestyle='--')
    plt.plot(x, bh_values, label='Market Buy & Hold (Baseline)', linewidth=2, color='#A8DADC', alpha=0.8)
    
    plt.title('Performance Comparison: Hybrid LSTM-RL vs Traditional Models', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Trading Days (Last 500)', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Text box metrics
    f_h, f_s, f_b = hybrid_values[-1], sma_values[-1], bh_values[-1]
    stats = (f"Hybrid: {((f_h-10000)/100):.1f}%\n"
             f"SMA:    {((f_s-10000)/100):.1f}%\n"
             f"B&H:    {((f_b-10000)/100):.1f}%")
    plt.text(0.02, 0.82, stats, transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    filename = "research_comparison_plot.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Success! Plot saved: {filename}")
    plt.show()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
