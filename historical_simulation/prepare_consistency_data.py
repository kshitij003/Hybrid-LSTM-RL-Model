import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import sys

# Add ml_service to path so we can import from it if needed
sys.path.append(os.path.join(os.getcwd(), 'ml_service'))

# Constants
STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
MODEL_NAME = "ProsusAI/finbert"

# Load FinBERT
print("[INFO] Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_sentiment(text):
    """Calculates sentiment score for a given text."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        probabilities = predictions[0].tolist()
        # FinBERT labels: 0 -> positive, 1 -> negative, 2 -> neutral
        sentiment_score = probabilities[0] - probabilities[1]
        return sentiment_score
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0.0

def main():
    raw_news_path = 'historical_simulation/data/consistency_raw_news.csv'
    if not os.path.exists(raw_news_path):
        print(f"[ERROR] {raw_news_path} not found. Run the scraper first.")
        return

    print("[INFO] Loading raw news...")
    df_news = pd.read_csv(raw_news_path)
    
    # Process sentiment
    print(f"[INFO] Processing sentiment for {len(df_news)} headlines. This may take a while...")
    df_news['sentiment'] = df_news['headline'].apply(get_sentiment)
    
    # Group by date and ticker
    daily_sentiment = df_news.groupby(['date', 'ticker'])['sentiment'].mean().reset_index()
    
    # Fetch price data roughly from early 2023 to late 2024
    start_date = datetime(2023, 1, 1) # Earlier for indicators
    end_date = datetime(2024, 7, 1)
    
    all_stock_data = []
    
    for ticker in STOCKS:
        print(f"[INFO] Fetching prices for {ticker}...")
        df_price = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        # Flatten index if needed (sometimes yfinance returns multi-index)
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = df_price.columns.get_level_values(0)
            
        df_price = df_price.reset_index()
        df_price['ticker'] = ticker
        df_price['Date'] = pd.to_datetime(df_price['Date']).dt.strftime('%Y-%m-%d')
        
        # Merge with sentiment
        stock_sentiment = daily_sentiment[daily_sentiment['ticker'] == ticker].copy()
        df_merged = pd.merge(df_price, stock_sentiment[['date', 'sentiment']], left_on='Date', right_on='date', how='left')
        
        # Fill missing sentiment with 0 (neutral)
        df_merged['sentiment'] = df_merged['sentiment'].fillna(0.0)
        
        # Drop extra date column
        if 'date' in df_merged.columns:
            df_merged = df_merged.drop(columns=['date'])
            
        all_stock_data.append(df_merged)
        
    final_df = pd.concat(all_stock_data)
    output_path = 'historical_simulation/data/consistency_dataset.csv'
    final_df.to_csv(output_path, index=False)
    print(f"[DONE] Consistency dataset saved to {output_path}")

if __name__ == "__main__":
    main()
