import os
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Handles downloading, caching, and returning market data.
    Supports Indian stocks via NSE suffix (e.g. RELIANCE.NS).
    """

    def __init__(self,
                 symbols=["RELIANCE.NS"],
                 start_date="2015-01-01",
                 end_date="2024-12-31",
                 interval="1d",
                 cache_file="cached_data.csv"):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.cache_file = cache_file

    # -----------------------------
    # Load CSV Manually (needed for train/backtest)
    # -----------------------------
    def load_csv(self, csv_path):
        logger.info(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.debug(f"Loaded {len(df)} rows from CSV")
        return df

    # -----------------------------
    def download_data(self):
        """
        Returns cached data if available, otherwise downloads fresh.
        """
        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached data from {self.cache_file}")
            df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
            logger.debug(f"Cached data loaded: {len(df)} rows")
            return df

        return self.fetch_and_cache()

    # -----------------------------
    def fetch_and_cache(self):
        """
        Downloads data from Yahoo Finance and caches it.
        Uses auto_adjust=True to handle Indian stock splits and bonus shares correctly.
        """
        logger.info(f"Fetching data for {self.symbols} from yfinance ({self.start_date} to {self.end_date})")

        data = yf.download(
            tickers=self.symbols,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=True,   # Essential for Indian stocks (splits, bonuses)
            progress=False,
            group_by='ticker'
        )
        
        logger.debug(f"Download complete. Raw shape: {data.shape}")

        # Fix MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # Remove prefix for single ticker
        if len(self.symbols) == 1:
            prefix = self.symbols[0] + "_"
            data.columns = [c.replace(prefix, "") for c in data.columns]

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        data.dropna(subset=required, inplace=True)
        logger.debug(f"Cleaned data: {len(data)} rows")

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        
        data.to_csv(self.cache_file)
        logger.info(f"Data cached to {self.cache_file}")

        return data
