import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import re

# Configuration
STOCKS = {
    "RELIANCE.NS": "RI",
    "TCS.NS": "TCS",
    "HDFCBANK.NS": "HDF01",
    "INFY.NS": "IT",
    "ICICIBANK.NS": "ICI02"
}

BASE_URL = "https://www.moneycontrol.com/stocks/company_info/stock_news.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
DATA_PATH = 'historical_simulation/data/raw_news.csv'
ARTICLES_PER_DAY_LIMIT = 5

def parse_mc_date(date_str):
    try:
        # Matches formats like '2.07 am | 09 Feb 2026' or '05.30 pm | 15 Jan 2025'
        parts = date_str.split('|')
        if len(parts) < 2: 
            # Try finding the date part with regex
            match = re.search(r'(\d{2} [A-Z][a-z]{2} \d{4})', date_str)
            if match:
                date_part = match.group(1)
            else:
                return None
        else:
            date_part = parts[1].strip()
            
        return datetime.strptime(date_part, "%d %b %Y")
    except:
        return None

def fetch_news_for_stock(ticker, sc_id, days_back=480):
    target_date = datetime.now() - timedelta(days=days_back)
    print(f"\n[INFO] Scraping news for {ticker} (Target: {target_date.date()})")
    
    current_year = datetime.now().year
    daily_counts = {} # date_str -> count
    
    for year in range(current_year, target_date.year - 1, -1):
        page = 1
        reached_target = False
        
        while not reached_target:
            url = f"{BASE_URL}?sc_id={sc_id}&durationType=Y&Year={year}&pageno={page}"
            print(f"   Page {page} for year {year}...", end='\r', flush=True)
            
            try:
                response = requests.get(url, headers=HEADERS, timeout=15)
                if response.status_code != 200:
                    print(f"\n   [WARN] Status {response.status_code} for {url}, stopping for this year.")
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Each news item is in a div with some specific padding/margin
                # Based on investigation, they often follow a pattern or are inside div.MT20
                news_blocks = soup.find_all('div', class_='MT20')
                if not news_blocks:
                    # Fallback to direct selection if container is different
                    headlines = soup.select('a.g_14bl')
                    date_tags = soup.select('p.PT10') or soup.select('p.a_10dgry')
                    items = []
                    for h, d in zip(headlines, date_tags):
                        items.append((h, d))
                else:
                    items = []
                    for block in news_blocks:
                        h = block.find('a', class_='g_14bl')
                        d = block.find('p', class_='PT10') or block.find('p', class_='a_10dgry') or block.find('p', class_='PT3')
                        if h and d:
                            items.append((h, d))

                if not items:
                    print(f"\n   [DEBUG] No news items found on page {page}")
                    break
                
                stock_news = []
                for h, d in items:
                    title = h.get_text(strip=True)
                    dt_str = d.get_text(strip=True)
                    dt_obj = parse_mc_date(dt_str)
                    
                    if dt_obj:
                        dt_iso = dt_obj.strftime('%Y-%m-%d')
                        if dt_obj < target_date:
                            reached_target = True
                            print(f"\n   [SUCCESS] Reached target date: {dt_obj.date()}")
                            break
                        
                        count = daily_counts.get(dt_iso, 0)
                        if count < ARTICLES_PER_DAY_LIMIT:
                            stock_news.append({
                                'ticker': ticker,
                                'date': dt_iso,
                                'headline': title
                            })
                            daily_counts[dt_iso] = count + 1
                
                if stock_news:
                    df_page = pd.DataFrame(stock_news)
                    df_page.to_csv(DATA_PATH, mode='a', header=not os.path.exists(DATA_PATH), index=False)
                    print(f"   [DEBUG] Saved {len(stock_news)} headlines to {DATA_PATH}", end='\r')
                
                if reached_target:
                    break
                    
                page += 1
                if page % 20 == 0:
                    time.sleep(1)
                
            except Exception as e:
                print(f"\n   [ERROR] On page {page}: {e}")
                break
                
    print(f"\n   Completed {ticker}")

def main():
    os.makedirs('historical_simulation/data', exist_ok=True)
    if os.path.exists(DATA_PATH):
        print("[INFO] Cleaning up previous raw_news.csv...")
        os.remove(DATA_PATH)

    for ticker, sc_id in STOCKS.items():
        fetch_news_for_stock(ticker, sc_id)
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"\n[DONE] Final dataset has {len(df)} headlines.")

if __name__ == "__main__":
    main()
