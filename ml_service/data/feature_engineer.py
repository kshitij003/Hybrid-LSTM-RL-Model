import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy.special
from sklearn.preprocessing import MinMaxScaler
from typing import List
import yfinance as yf


class FinBERTAnalyzer:
    """
    A class to load and use the FinBERT model for sentiment analysis.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        logger.info(f"Loading FinBERT model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            self.tokenizer = None
            self.model = None

    def get_sentiment_scores(self, texts: List[str]) -> List[float]:
        if self.model is None or self.tokenizer is None:
            return [0.0] * len(texts)

        try:
            inputs = self.tokenizer(
                texts, padding=True, truncation=True,
                return_tensors='pt', max_length=512
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = scipy.special.softmax(outputs.logits.numpy(), axis=1)
            sentiment_scores = (probs[:, 0] - probs[:, 1]).tolist()
            return sentiment_scores

        except Exception as e:
            logger.warning(f"FinBERT inference error: {e}")
            return [0.0] * len(texts)


class FeatureEngineer:
    """
    Applies technical indicators, sentiment, and SAFE normalization.
    """
    def __init__(self):
        self.scalers = {}

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Adding technical indicators (RSI, MACD) to {len(df)} rows...")

        # pandas_ta requires simple column names and a valid close price
        if "Close" not in df.columns:
            raise ValueError("The dataframe does not contain a 'Close' column required for technical indicators.")

        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)

        # Safe fill (pandas 2.x compatible)
        df = df.replace([np.inf, -np.inf], np.nan)
        df.bfill(inplace=True)
        df.ffill(inplace=True)

        return df

    def add_sentiment_data(
        self,
        df: pd.DataFrame,
        sentiment_scores: list = None,
    ) -> pd.DataFrame:
        """
        Attach a sentiment column to df.

        Args:
            df: Feature DataFrame (index = date or integer)
            sentiment_scores: Optional list of floats (real FinBERT scores, one per row).
                              If None or empty, falls back to a simulated sine-wave signal
                              and logs a warning so it is visible in training logs.
        """
        if sentiment_scores and len(sentiment_scores) == len(df):
            logger.debug("   ✅ Using real FinBERT sentiment scores for training.")
            df["sentiment"] = sentiment_scores
        else:
            if sentiment_scores is not None:
                logger.warning(
                    f"   ⚠️  Sentiment score length mismatch "
                    f"(got {len(sentiment_scores)}, need {len(df)}). "
                    "Falling back to simulated signal."
                )
            else:
                logger.warning(
                    "   ⚠️  No real sentiment scores provided — using simulated signal. "
                    "Pass sentiment_scores for FinBERT-backed training."
                )
            np.random.seed(42)
            random_walk = np.random.randn(len(df)).cumsum()
            df["sentiment"] = np.sin(random_walk / 50) + np.random.normal(0, 0.1, len(df))

        return df

    def fetch_and_score_sentiment(
        self,
        ticker: str,
        df_index: pd.Index,
        news_api_key: str = None,
        days_back: int = 30,
    ):
        """
        Fetches recent news for `ticker` via NewsAPI and scores each headline
        with FinBERT. Returns a list of floats aligned to df_index, or None
        if real scores cannot be obtained.

        Returning None (not zeros) is intentional — the caller (add_sentiment_data)
        will then fall back to a consistent simulated signal. Returning zeros would
        teach the LSTM that 'data unavailable' equals 'perfectly neutral market',
        which corrupts the model and causes misleading trade signals.

        Args:
            ticker:       NSE/BSE/US ticker e.g. 'RELIANCE.NS'
            df_index:     DatetimeIndex of the feature DataFrame to align scores to
            news_api_key: NewsAPI key (reads NEWS_API_KEY env var if None)
            days_back:    How many calendar days of news to fetch
        Returns:
            list[float] of length len(df_index), or None on any failure
        """
        import os
        import requests
        from datetime import datetime, timedelta

        api_key = news_api_key or os.getenv("NEWS_API_KEY", "")
        if not api_key or api_key in ("YOUR_API_KEY_HERE", ""):
            logger.debug(f"   ⚠️  NEWS_API_KEY not set — will use simulated sentiment for {ticker}.")
            return None   # ← triggers consistent simulated fallback, NOT zeros

        try:
            COMPANY_MAP = {
                'RELIANCE': 'Reliance Industries', 'TCS': 'Tata Consultancy Services',
                'HDFCBANK': 'HDFC Bank',           'INFY': 'Infosys',
                'ICICIBANK': 'ICICI Bank',          'WIPRO': 'Wipro',
                'HCLTECH': 'HCL Technologies',      'AXISBANK': 'Axis Bank',
                'SBIN': 'State Bank of India',      'BAJFINANCE': 'Bajaj Finance',
                'BHARTIARTL': 'Bharti Airtel',      'KOTAKBANK': 'Kotak Mahindra Bank',
                'TATAMOTORS': 'Tata Motors',        'TATASTEEL': 'Tata Steel',
                'MARUTI': 'Maruti Suzuki',           'SUNPHARMA': 'Sun Pharmaceutical',
                'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google',
                'AMZN': 'Amazon', 'TSLA': 'Tesla',
            }
            base    = ticker.split('.')[0].upper()
            company = COMPANY_MAP.get(base, base)
            suffix  = ' NSE India' if '.NS' in ticker or '.BO' in ticker else ''
            query   = f"{company} stock{suffix}"

            end_dt   = datetime.now()
            start_dt = end_dt - timedelta(days=days_back)

            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    'q': query, 'language': 'en', 'sortBy': 'publishedAt',
                    'from': start_dt.strftime('%Y-%m-%d'),
                    'to':   end_dt.strftime('%Y-%m-%d'),
                    'apiKey': api_key, 'pageSize': 50,
                },
                timeout=10,
            )

            if resp.status_code != 200:
                logger.warning(f"   ⚠️  NewsAPI {resp.status_code} for {ticker} — using simulated sentiment.")
                return None   # ← NOT zeros

            articles = resp.json().get('articles', [])
            if not articles:
                print(f"   ⚠️  No news articles for {ticker} — using simulated sentiment.")
                return None   # ← NOT zeros

            # Score headlines with FinBERT
            finbert = FinBERTAnalyzer()
            date_scores: dict = {}   # 'YYYY-MM-DD' -> list[float]
            for art in articles:
                pub  = art.get('publishedAt', '')[:10]
                text = f"{art.get('title', '')} {art.get('description', '')}".strip()
                if text:
                    s = finbert.get_sentiment_scores([text])
                    date_scores.setdefault(pub, []).append(s[0])

            if not date_scores:
                print(f"   ⚠️  FinBERT scored 0 articles for {ticker} — using simulated sentiment.")
                return None   # ← NOT zeros

            # Forward-fill across df_index.
            # Days before the first scored article stay at 0.0 (genuinely unknown,
            # but this is historical data so it's only the tail-end of a long series).
            aligned, last_score = [], 0.0
            for idx in df_index:
                date_str = str(idx)[:10]
                if date_str in date_scores:
                    last_score = float(np.mean(date_scores[date_str]))
                aligned.append(last_score)

            print(f"   ✅ Real FinBERT sentiment aligned for {ticker} ({len(articles)} articles, "
                  f"{len(date_scores)} scored days).")
            return aligned

        except Exception as e:
            logger.error(f"   ❌ fetch_and_score_sentiment failed for {ticker}: {e} — using simulated sentiment.")
            return None   # ← NOT zeros


    def safe_minmax(self, series: pd.Series) -> pd.Series:
        """Safe MinMaxScaler preventing division by zero."""
        min_v = series.min()
        max_v = series.max()

        if pd.isna(min_v) or pd.isna(max_v):
            return series.fillna(0)

        # If constant column → return mid-value (0.5)
        if max_v - min_v == 0:
            return series.apply(lambda x: 0.5)

        return (series - min_v) / (max_v - min_v)

    def normalize_data(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        logger.debug(f"Normalizing features safely: {feature_cols}")
        df_norm = df.copy()

        for col in feature_cols:
            df_norm[col] = self.safe_minmax(df_norm[col])
            self.scalers[col] = (df[col].min(), df[col].max())

        df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
        df_norm.bfill(inplace=True)
        df_norm.ffill(inplace=True)
        df_norm.fillna(0, inplace=True)

        return df_norm

    def get_feature_df(
        self,
        df_raw: pd.DataFrame,
        sentiment_scores: list = None,
    ):
        """
        Full feature pipeline.

        Args:
            df_raw:            Raw OHLCV DataFrame from yfinance.
            sentiment_scores:  Optional list of real FinBERT scores (one per row).
                               Pass the output of fetch_and_score_sentiment() here
                               during training to use real news data.
                               If None, a simulated signal is used.
        Returns:
            (df_normalised, feature_cols)
        """
        # --- FIX MULTI-INDEX ---
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = ["_".join(col).strip() for col in df_raw.columns]

        # --- ENSURE CLOSE COLUMN EXISTS ---
        if "Adj Close" in df_raw.columns:
            df_raw["Close"] = df_raw["Adj Close"]
        elif "Close" not in df_raw.columns:
            possible = [c for c in df_raw.columns if "close" in c.lower()]
            if not possible:
                raise ValueError("No valid Close or Adj Close column found.")
            df_raw["Close"] = df_raw[possible[0]]

        # Add technical indicators
        df = self.add_technical_indicators(df_raw.copy())

        # Add sentiment (real FinBERT scores if provided, else simulated)
        df = self.add_sentiment_data(df, sentiment_scores=sentiment_scores)

        # Numeric feature columns (exclude Volume)
        feature_cols = [
            col for col in df.columns
            if df[col].dtype != "object" and col not in ["Volume"]
        ]

        df_norm = self.normalize_data(df, feature_cols)
        return df_norm[feature_cols].dropna(), feature_cols


if __name__ == "__main__":
    analyzer = FinBERTAnalyzer()
    headlines = [
        "Reliance Industries posts record quarterly profit.",
        "RBI raises interest rates amid inflation concerns.",
        "Infosys raises FY25 revenue guidance on strong deal wins."
    ]
    scores = analyzer.get_sentiment_scores(headlines)

    print("\nFinBERT Sentiment Scores (Indian Market Headlines):")
    for h, s in zip(headlines, scores):
        print(f"{s:.4f} \u2192 {h}")

    import yfinance as yf
    print("\nDownloading sample RELIANCE.NS data...")
    sample_df = yf.download("RELIANCE.NS", start="2024-01-01", end="2024-06-01", auto_adjust=True)

    fe = FeatureEngineer()
    features_df, cols = fe.get_feature_df(sample_df)   # simulated sentiment

    print("\nProcessed Features (simulated sentiment):")
    print(features_df.head())
    print("\nFeature Columns:", cols)
