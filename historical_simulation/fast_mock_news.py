import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

def create_mock_news():
    print("Generating mock news based on existing headlines...")
    raw_news_path = 'historical_simulation/data/raw_news.csv'
    
    df_existing = pd.read_csv(raw_news_path)
    headlines = df_existing['headline'].dropna().unique().tolist()
    
    STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=480)
    
    new_rows = []
    
    for ticker in STOCKS:
        # Generate ~200 news items per stock
        for _ in range(200):
            random_days = random.randint(0, 480)
            random_date = start_date + timedelta(days=random_days)
            random_headline = random.choice(headlines)
            
            ticker_name = ticker.split('.')[0]
            if ticker_name == 'RELIANCE':
                name = 'Reliance'
            elif ticker_name == 'TCS':
                name = 'TCS'
            elif ticker_name == 'INFY':
                name = 'Infosys'
            elif ticker_name == 'HDFCBANK':
                name = 'HDFC'
            else:
                name = 'ICICI'
                
            hl = random_headline.replace('Reliance', name).replace('RIL', name).replace('ONGC', 'competitor')
            
            new_rows.append({
                'ticker': ticker,
                'date': random_date.strftime('%Y-%m-%d'),
                'headline': hl
            })
            
    df_all = pd.DataFrame(new_rows)
    df_all = df_all.sort_values(by=['date'])
    df_all.to_csv(raw_news_path, index=False)
    print(f"Generated {len(df_all)} headlines and saved to {raw_news_path}")

if __name__ == "__main__":
    create_mock_news()
