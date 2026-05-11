import pandas as pd
from gnews import GNews
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import sys
import io
import time

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def fetch_historical_news():
    print("="*80)
    print("🚀 STARTING REAL HISTORICAL NEWS SCRAPER (CONSISTENCY TEST)")
    print("="*80)

    STOCKS = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys",
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank"
    }

    # Consistency dataset runs roughly 480 trading days prior to the original simulation
    # (e.g., from mid-2023 to late-2024)
    start_date = datetime(2023, 3, 1)
    end_date = datetime(2024, 7, 1)

    all_news = []
    
    for ticker, company in STOCKS.items():
        print(f"\n📰 Fetching news for {ticker} ({company})...")
        
        current_date = start_date
        while current_date < end_date:
            next_month = current_date + relativedelta(months=1)
            
            # gnews expects dates as tuples (YYYY, MM, DD)
            start_tuple = (current_date.year, current_date.month, current_date.day)
            end_tuple = (next_month.year, next_month.month, next_month.day)
            
            google_news = GNews(start_date=start_tuple, end_date=end_tuple, max_results=100)
            
            query = f'"{company}" NSE'
            try:
                articles = google_news.get_news(query)
                print(f"   [{current_date.strftime('%Y-%m')}] Found {len(articles)} articles.")
                
                for art in articles:
                    try:
                        # gnews returns published date like 'Tue, 16 Jan 2024 08:00:00 GMT'
                        pub_date = pd.to_datetime(art['published date']).strftime('%Y-%m-%d')
                        title = art['title']
                        
                        all_news.append({
                            'date': pub_date,
                            'ticker': ticker,
                            'headline': title
                        })
                    except Exception as e:
                        continue
            except Exception as e:
                print(f"   [{current_date.strftime('%Y-%m')}] Error: {e}")
                
            current_date = next_month
            time.sleep(1) # Be nice to Google
            
    df = pd.DataFrame(all_news)
    
    if len(df) > 0:
        # Drop duplicates in case of overlap
        df = df.drop_duplicates(subset=['date', 'ticker', 'headline'])
        
        # Sort
        df = df.sort_values(by=['date', 'ticker'])
        
        os.makedirs('historical_simulation/data', exist_ok=True)
        out_path = 'historical_simulation/data/consistency_raw_news.csv'
        df.to_csv(out_path, index=False)
        
        print("\n" + "="*80)
        print(f"✅ Scraping complete! Saved {len(df)} real historical headlines to {out_path}.")
        print("="*80)
    else:
        print("\n❌ Failed to scrape any news.")

if __name__ == "__main__":
    fetch_historical_news()
