import pandas as pd
import numpy as np
from data.data_handler import DataHandler
from data.feature_engineer import FeatureEngineer
from datetime import datetime, timedelta

def audit():
    STOCKS = ["RELIANCE.NS", "TCS.NS"]
    fe = FeatureEngineer()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)
    
    for ticker in STOCKS:
        print(f"\n--- AUDIT: {ticker} ---")
        handler = DataHandler(symbols=[ticker], start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
        df = handler.download_data()
        
        # Look at raw prices
        print("Raw Prices (last 5 days):")
        print(df['Close'].tail())
        
        # Look at engineered features
        df_feat, _ = fe.get_feature_df(df)
        df_feat = df_feat.dropna()
        
        print("\nEngineered Features (last 5 rows):")
        cols_to_show = ['Close', 'RSI_14', 'MACD_12_26_9']
        print(df_feat[cols_to_show].tail())
        
        if df_feat['Close'].max() < 10.0:
            print("\n WARNING: Prices seem to be NORMALIZED (0-1). Trading with ₹10,000 cash will fail!")

if __name__ == "__main__":
    audit()
