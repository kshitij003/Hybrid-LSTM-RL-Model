"""
Module 2: Advanced Feature Engineering
Includes FinBERT for sentiment analysis and technical indicators.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
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
        print("Loading FinBERT model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            print("FinBERT model loaded successfully.")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
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
            print(f"FinBERT inference error: {e}")
            return [0.0] * len(texts)


class FeatureEngineer:
    """
    A class to apply feature engineering to OHLCV data.
    """
    def __init__(self):
        self.scalers = {}

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding technical indicators (RSI, MACD)...")

        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)

        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        return df

    def add_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding simulated sentiment signal...")
        np.random.seed(42)
        random_walk = np.random.randn(len(df)).cumsum()
        sentiment_signal = np.sin(random_walk / 50) + np.random.normal(0, 0.1, len(df))
        df["sentiment"] = sentiment_signal
        return df

    def normalize_data(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        print(f"Normalizing features: {feature_cols}")
        df_norm = df.copy()

        for col in feature_cols:
            scaler = MinMaxScaler((0, 1))
            df_norm[col] = scaler.fit_transform(df_norm[col].values.reshape(-1, 1))
            self.scalers[col] = scaler

        return df_norm

    def get_feature_df(self, df_raw: pd.DataFrame) -> (pd.DataFrame, List[str]):
        df = self.add_technical_indicators(df_raw.copy())
        df = self.add_sentiment_data(df)

        df["Close"] = df["Adj Close"]  # Force Close = Adj Close for consistency

        # Automatically detect numeric feature columns
        feature_cols = [
            col for col in df.columns
            if df[col].dtype != "object" and col not in ["Volume"]
        ]

        # Normalize features
        df_norm = self.normalize_data(df, feature_cols)

        return df_norm[feature_cols].dropna(), feature_cols


if __name__ == "__main__":
    # Example FinBERT usage
    analyzer = FinBERTAnalyzer()
    headlines = [
        "Apple stock rallies after record earnings.",
        "Federal Reserve warns of possible recession.",
        "Markets remain stable amid global tensions."
    ]
    scores = analyzer.get_sentiment_scores(headlines)

    print("\nFinBERT Sentiment Scores:")
    for h, s in zip(headlines, scores):
        print(f"{s:.4f} → {h}")

    # Example feature engineering usage
    print("\nDownloading sample MSFT data...")
    sample_df = yf.download("MSFT", start="2020-01-01", end="2020-06-01")

    fe = FeatureEngineer()
    features_df, cols = fe.get_feature_df(sample_df)

    print("\nProcessed Features:")
    print(features_df.head())
    print("\nFeature Columns:", cols)
    print("\nScalers:", list(fe.scalers.keys()))
