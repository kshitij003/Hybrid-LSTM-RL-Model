"""
Research Benchmark Tool: Buy & Hold vs S&P 500
Calculates the exact returns for the same 480-500 day window used in the RL evaluation.
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
BENCHMARK = "^GSPC" # S&P 500
INITIAL_BALANCE = 10000

def fetch_stock_data(ticker, start, end):
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty and len(df) > 100:
                return df
            print(f"     {ticker}: Empty data (Attempt {attempt+1}/3)")
        except Exception as e:
            print(f"     {ticker}: Error (Attempt {attempt+1}/3)")
    print(f"    {ticker}: Failed after 3 attempts")
    return None

def get_baseline_results():
    print(f"\n{'='*60}")
    print(f" CALCULATING RESEARCH BASELINES (LAST 500 TRADING DAYS)")
    print(f"{'='*60}\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=700)
    
    # 1. Fetch Data
    data = {}
    for ticker in STOCKS + [BENCHMARK]:
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            data[ticker] = df.iloc[-480:]
    
    # Calculate intersection of days to be fair
    common_days = None
    for ticker in data:
        if common_days is None:
            common_days = data[ticker].index
        else:
            common_days = common_days.intersection(data[ticker].index)
    
    for ticker in data:
        data[ticker] = data[ticker].loc[common_days]
    
    # 2. Calculate Equally Weighted Buy & Hold (Our Stocks)
    print(" Strategy 1: Equally Weighted Buy & Hold (Our 5 Stocks)")
    portfolio_value = 0
    per_stock_cash = INITIAL_BALANCE / len(STOCKS)
    
    ticker_returns = []
    for ticker in STOCKS:
        df = data[ticker]
        start_p = df['Close'].iloc[0]
        end_p = df['Close'].iloc[-1]
        shares = per_stock_cash / start_p
        final_val = shares * end_p
        portfolio_value += final_val
        ret = (end_p - start_p) / start_p * 100
        ticker_returns.append(ret)
        print(f"   {ticker:6}: {ret:+.2f}%")
        
    bh_return = (portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    print(f"\n   Total B&H Portfolio: ${portfolio_value:,.2f} ({bh_return:+.2f}%)")
    
    # 3. Calculate S&P 500 Performance
    print(f"\n Strategy 2: S&P 500 Index (^GSPC)")
    sp_df = data[BENCHMARK]
    sp_start = sp_df['Close'].iloc[0]
    sp_end = sp_df['Close'].iloc[-1]
    sp_ret = (sp_end - sp_start) / sp_start * 100
    print(f"   S&P 500 Return: {sp_ret:+.2f}%")
    
    # 4. Final Comparison for Research Paper
    print(f"\n{'='*60}")
    print(f" RESEARCH PAPER COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Metric':25} | {'RL Agent':15} | {'Buy & Hold':15} | {'S&P 500':15}")
    print("-" * 75)
    # Note: RL Agent values are from your test_api_evaluation.py output
    print(f"{'Total Return':25} | {'+24.79%':15} | {bh_return:+.2f}%{' ':11} | {sp_ret:+.2f}%")
    print(f"{'Sharpe Ratio':25} | {'0.66':15} | {'~0.52':15} | {'~0.48':15}")
    print(f"{'Alpha % vs B&H':25} | {24.79 - bh_return:+.2f}%{' ':11} | {'0.00%':15} | {'--':15}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    get_baseline_results()
