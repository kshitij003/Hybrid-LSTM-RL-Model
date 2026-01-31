import os
import pandas as pd
import yfinance as yf


class DataHandler:
    """
    Handles downloading, caching, and returning market data.
    """

    def __init__(self,
                 symbols=["AAPL"],
                 start_date="2010-01-01",
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
        print(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    # -----------------------------
    def download_data(self):
        """
        Returns cached data if available, otherwise downloads fresh.
        """
        if os.path.exists(self.cache_file):
            print("Loading cached data...")
            return pd.read_csv(self.cache_file, index_col=0, parse_dates=True)

        return self.fetch_and_cache()

    # -----------------------------
    def fetch_and_cache(self):
        """
        Downloads data from Yahoo Finance and caches it.
        """
        print(f"Fetching data for {self.symbols} from yfinance...")

        data = yf.download(
            tickers=self.symbols,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=False,
            progress=False,
            group_by='ticker'
        )

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

        data.to_csv(self.cache_file)

        return data
