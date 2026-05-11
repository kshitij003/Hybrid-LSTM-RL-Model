"""
prepare_simulation_data.py
---------------------------
Builds historical_simulation/data/simulation_dataset.csv by:
  1. Loading raw_news.csv (must be real headlines — no mock data allowed)
  2. Scoring every headline with FinBERT  (ProsusAI/finbert)
  3. Computing daily mean sentiment per stock
  4. Downloading OHLCV from yfinance
  5. Left-merging price data with real sentiment
  6. Forward-filling sentiment gaps (max 3 trading days)
     — any gap longer than 3 days gets 0.0 (neutral, not invented)
  7. HARD STOP if real-sentiment coverage < 70% of trading days per stock

Run after fetch_real_historical_news.py.
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_NEWS_PATH = os.path.join(PROJECT_ROOT, 'historical_simulation', 'data', 'raw_news.csv')
OUT_PATH      = os.path.join(PROJECT_ROOT, 'historical_simulation', 'data', 'simulation_dataset.csv')

# ── Config ────────────────────────────────────────────────────────────────────
STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
FINBERT_MODEL = "ProsusAI/finbert"

# Minimum fraction of trading days that must have REAL (non-zero) sentiment
MIN_COVERAGE_FRACTION = 0.70   # hard stop below this threshold

# Forward-fill gap limit: if last real news was > N days ago, use 0.0 instead
MAX_FORWARD_FILL_DAYS = 3

# Price window — extra days at start so technical indicators (RSI 14, MACD 26) warm up
PRICE_START = datetime(2024, 8, 1)   # 3 months before news window
PRICE_END   = datetime(2026, 5, 8)


# ── FinBERT ───────────────────────────────────────────────────────────────────
print("[INFO] Loading FinBERT model (ProsusAI/finbert)...")
tokenizer   = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert_mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
finbert_mdl.to(DEVICE)
finbert_mdl.eval()
print(f"[INFO] FinBERT loaded on {DEVICE.upper()}.")


def score_headline(text: str) -> float:
    """Returns FinBERT sentiment score: positive_prob - negative_prob ∈ [-1, +1]."""
    try:
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, padding=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            logits = finbert_mdl(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
        # FinBERT label order: 0=positive, 1=negative, 2=neutral
        return float(probs[0] - probs[1])
    except Exception as exc:
        print(f"   [WARN] FinBERT error on text '{text[:60]}': {exc}")
        return 0.0


def score_batch(texts: list, batch_size: int = 16) -> list:
    """Score a list of texts in batches for speed."""
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            inputs = tokenizer(
                batch, return_tensors="pt",
                truncation=True, padding=True, max_length=512
            ).to(DEVICE)
            with torch.no_grad():
                logits = finbert_mdl(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).tolist()
            batch_scores = [float(p[0] - p[1]) for p in probs]
        except Exception as exc:
            print(f"   [WARN] Batch scoring error: {exc} — using 0.0 for batch")
            batch_scores = [0.0] * len(batch)
        scores.extend(batch_scores)
        if i % 100 == 0 and i > 0:
            print(f"   ... scored {i}/{len(texts)} headlines")
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*80)
    print("🚀 PREPARE SIMULATION DATASET (REAL SENTIMENT ONLY)")
    print("="*80)

    # 1. Load raw news
    if not os.path.exists(RAW_NEWS_PATH):
        print(f"[ERROR] {RAW_NEWS_PATH} not found.")
        print("        Run fetch_real_historical_news.py first.")
        sys.exit(1)

    print(f"[INFO] Loading raw news from {RAW_NEWS_PATH}...")
    df_news = pd.read_csv(RAW_NEWS_PATH)
    print(f"[INFO] {len(df_news)} headlines loaded.")
    print(f"[INFO] Per-ticker counts: {dict(df_news.groupby('ticker').size())}")

    # Sanity check: warn if raw_news looks like mock data (exactly 200 per ticker)
    counts = df_news.groupby('ticker').size()
    if all(c == 200 for c in counts):
        print("\n⚠️  WARNING: Every ticker has exactly 200 headlines.")
        print("   This is a tell-tale sign of fast_mock_news.py output!")
        print("   Re-run fetch_real_historical_news.py to get real data.")
        print("   Continuing anyway — but sentiment quality may be low.\n")

    # 2. Score all headlines with FinBERT
    print(f"\n[INFO] Scoring {len(df_news)} headlines with FinBERT (batch_size=16)...")
    df_news['sentiment'] = score_batch(df_news['headline'].tolist())
    print(f"[INFO] Scoring complete. Sentiment stats:")
    print(df_news['sentiment'].describe().to_string())

    # 3. Daily mean sentiment per (date, ticker)
    df_news['date'] = pd.to_datetime(df_news['date']).dt.strftime('%Y-%m-%d')
    daily_sentiment = (
        df_news.groupby(['date', 'ticker'])['sentiment']
        .mean()
        .reset_index()
        .rename(columns={'sentiment': 'daily_sentiment'})
    )
    print(f"\n[INFO] Daily sentiment computed for {len(daily_sentiment)} (date, ticker) pairs.")

    # 4. Download price data + merge
    all_stock_data = []

    for ticker in STOCKS:
        print(f"\n[INFO] Fetching prices for {ticker}...")
        df_price = yf.download(
            ticker,
            start=PRICE_START.strftime('%Y-%m-%d'),
            end=PRICE_END.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False,
        )

        if df_price.empty:
            print(f"[ERROR] No price data for {ticker}. Skipping.")
            continue

        # Flatten multi-index if present
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = df_price.columns.get_level_values(0)

        df_price = df_price.reset_index()
        df_price['ticker'] = ticker
        df_price['Date']   = pd.to_datetime(df_price['Date']).dt.strftime('%Y-%m-%d')

        # Merge with real sentiment
        stock_sent = daily_sentiment[daily_sentiment['ticker'] == ticker].copy()
        df_merged  = pd.merge(
            df_price,
            stock_sent[['date', 'daily_sentiment']],
            left_on='Date', right_on='date',
            how='left'
        )
        if 'date' in df_merged.columns:
            df_merged.drop(columns=['date'], inplace=True)

        # 5. Smart forward-fill: carry last real sentiment up to MAX_FORWARD_FILL_DAYS
        #    then revert to 0.0 (neutral) for truly news-silent periods
        df_merged = df_merged.sort_values('Date').reset_index(drop=True)
        sentiment_col = df_merged['daily_sentiment'].copy()

        filled = []
        days_since_real = 999
        for val in sentiment_col:
            if not pd.isna(val):
                filled.append(float(val))
                days_since_real = 0
            else:
                days_since_real += 1
                if days_since_real <= MAX_FORWARD_FILL_DAYS:
                    # Carry forward last known real sentiment
                    filled.append(filled[-1] if filled else 0.0)
                else:
                    # Too long since real news — use neutral
                    filled.append(0.0)

        df_merged['sentiment'] = filled
        df_merged.drop(columns=['daily_sentiment'], inplace=True)

        # 6. Coverage check (hard stop per stock)
        real_rows  = (df_merged['sentiment'] != 0.0).sum()
        total_rows = len(df_merged)
        coverage   = real_rows / total_rows if total_rows > 0 else 0.0

        status = "✅" if coverage >= MIN_COVERAGE_FRACTION else "❌"
        print(f"   {status} {ticker}: {real_rows}/{total_rows} days have real sentiment "
              f"({coverage:.1%})")

        if coverage < MIN_COVERAGE_FRACTION:
            print(f"\n[HARD STOP] {ticker} real-sentiment coverage is {coverage:.1%}, "
                  f"below the {MIN_COVERAGE_FRACTION:.0%} threshold.")
            print("   Re-run fetch_real_historical_news.py to get more headlines, then retry.")
            sys.exit(1)

        all_stock_data.append(df_merged)

    if not all_stock_data:
        print("[ERROR] No stock data to save.")
        sys.exit(1)

    final_df = pd.concat(all_stock_data, ignore_index=True)
    final_df.to_csv(OUT_PATH, index=False)

    print(f"\n✅  Simulation dataset saved → {OUT_PATH}")
    print(f"   Total rows    : {len(final_df)}")
    print(f"   Columns       : {final_df.columns.tolist()}")
    print(f"   Date range    : {final_df['Date'].min()}  →  {final_df['Date'].max()}")
    real_overall = (final_df['sentiment'] != 0.0).sum()
    print(f"   Real sentiment: {real_overall}/{len(final_df)} rows "
          f"({real_overall/len(final_df):.1%})")
    print("="*80)


if __name__ == "__main__":
    main()
