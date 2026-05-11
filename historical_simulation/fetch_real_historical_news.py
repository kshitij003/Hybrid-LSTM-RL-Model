"""
fetch_real_historical_news.py
------------------------------
Fetches REAL per-day, per-stock Google News headlines using the gnews library.
- Month-by-month window to maximise coverage
- Exponential backoff retry on transient failures
- Deduplicates headlines within the same day/ticker
- Saves to historical_simulation/data/raw_news.csv
- NEVER uses mock/dummy/shuffled data — crashes explicitly if GNews fails 3+ times
"""

import os
import sys
import io
import time
import random
import pandas as pd
from gnews import GNews
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Configuration ─────────────────────────────────────────────────────────────

STOCKS = {
    "RELIANCE.NS":  "Reliance Industries",
    "TCS.NS":       "Tata Consultancy Services",
    "INFY.NS":      "Infosys",
    "HDFCBANK.NS":  "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
}

# Full backtest window (must match simulation_dataset date range)
START_DATE = datetime(2024, 11, 1)
END_DATE   = datetime(2026, 5, 8)   # today-ish

# Scrape up to N articles per month-window per stock
MAX_ARTICLES_PER_WINDOW = 100

# Retry settings for GNews failures
MAX_RETRIES   = 3
BACKOFF_BASE  = 2   # seconds (doubles each retry)

# Output path (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, 'historical_simulation', 'data', 'raw_news.csv')


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_month_with_retry(company: str, start_tup: tuple, end_tup: tuple) -> list:
    """
    Fetch articles for one month window with exponential backoff.
    Returns a list of article dicts, or raises RuntimeError after MAX_RETRIES.
    """
    queries = [
        f'"{company}" NSE stock',
        f'"{company}" share price India',
        f'"{company}"',
    ]

    last_exc = None
    for attempt in range(MAX_RETRIES):
        # Rotate query slightly on retries to avoid cached empty results
        query = queries[attempt % len(queries)]
        try:
            gn = GNews(start_date=start_tup, end_date=end_tup,
                       max_results=MAX_ARTICLES_PER_WINDOW)
            articles = gn.get_news(query)
            if articles is None:
                articles = []
            return articles
        except Exception as exc:
            last_exc = exc
            wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
            print(f"      [Retry {attempt+1}/{MAX_RETRIES}] Error: {exc}  → waiting {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(
        f"GNews failed {MAX_RETRIES} times for '{company}' "
        f"window {start_tup}–{end_tup}: {last_exc}"
    )


def parse_article(art: dict, ticker: str) -> dict | None:
    """Extract date + headline from a GNews article dict. Returns None to skip."""
    try:
        pub_date = pd.to_datetime(art['published date']).strftime('%Y-%m-%d')
        title    = str(art.get('title', '')).strip()
        if not title:
            return None
        return {'date': pub_date, 'ticker': ticker, 'headline': title}
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_historical_news():
    print("\n" + "="*80)
    print("🚀 STARTING REAL HISTORICAL NEWS FETCH (NO DUMMIES)")
    print(f"   Period : {START_DATE.date()} → {END_DATE.date()}")
    print(f"   Stocks : {', '.join(STOCKS.keys())}")
    print("="*80)

    all_rows    = []
    fail_months = []   # track (ticker, month) windows that returned 0 articles

    for ticker, company in STOCKS.items():
        print(f"\n📰  {ticker}  ({company})")
        current = START_DATE

        while current < END_DATE:
            next_month = current + relativedelta(months=1)
            label      = current.strftime('%Y-%m')

            start_tup = (current.year,    current.month,    current.day)
            end_tup   = (next_month.year, next_month.month, next_month.day)

            try:
                articles = fetch_month_with_retry(company, start_tup, end_tup)
            except RuntimeError as e:
                print(f"   ❌  [{label}] FAILED after {MAX_RETRIES} retries: {e}")
                fail_months.append((ticker, label))
                current = next_month
                time.sleep(2)
                continue

            parsed_count = 0
            for art in articles:
                row = parse_article(art, ticker)
                if row:
                    all_rows.append(row)
                    parsed_count += 1

            print(f"   [{label}]  {parsed_count} real headlines")

            if parsed_count == 0:
                fail_months.append((ticker, label))

            current = next_month
            time.sleep(1.5)   # polite delay between requests

    # ── Post-processing ──────────────────────────────────────────────────────
    if not all_rows:
        print("\n❌  No articles fetched at all — aborting. Do NOT fall back to mock data.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=['date', 'ticker', 'headline'])
    print(f"\n   Dedup: {before} → {len(df)} rows")

    # Sort
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Coverage report
    print("\n📊  Coverage Report:")
    print(f"   Total headlines : {len(df)}")
    print(f"   Date range      : {df['date'].min()}  →  {df['date'].max()}")
    print("   Per-ticker counts:")
    for t, cnt in df.groupby('ticker').size().items():
        print(f"      {t:20s}  {cnt:5d} headlines")

    if fail_months:
        print(f"\n⚠️   {len(fail_months)} month-windows returned 0 articles:")
        for t, m in fail_months:
            print(f"      {t}  {m}")
        print("   These will get neutral (0.0) sentiment during prepare_simulation_data.py")

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n✅  Saved {len(df)} real headlines to {OUT_PATH}")
    print("="*80)
    return df


if __name__ == "__main__":
    fetch_historical_news()
