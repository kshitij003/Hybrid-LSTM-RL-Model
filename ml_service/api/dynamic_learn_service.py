"""
Dynamic Learning Service
Handles on-the-fly LSTM training for new tickers
"""

from flask import Blueprint, request, jsonify
import threading
import yfinance as yf
import pandas as pd
import os
from models.lstm_model import train_lstm_model
from data.feature_engineer import FeatureEngineer

dynamic_learn_bp = Blueprint('dynamic_learn', __name__)

# Dictionary to track ongoing training tasks
training_status = {}

def train_new_ticker_task(ticker):
    try:
        training_status[ticker] = "DOWNLOADING_DATA"
        print(f" Downloading data for {ticker}...")
        
        # 1. Fetch data from yfinance
        df = yf.download(ticker, period="2y", interval="1d")
        if df.empty:
            training_status[ticker] = "FAILED: No data found"
            return

        # 2. Add features
        training_status[ticker] = "ENGINEERING_FEATURES"
        fe = FeatureEngineer()
        df_features, feature_cols = fe.get_feature_df(df)
        
        # 3. Train LSTM
        training_status[ticker] = "TRAINING_LSTM"
        model_path = f"models/saved_models/lstm_{ticker}.pth"
        
        # Use default hyperparameters for consistency
        train_lstm_model(
            train_df=df_features,
            feature_cols=feature_cols,
            target_col='Close',
            sequence_length=30,
            epochs=15, # Faster for dynamic training during demo
            model_save_path=model_path
        )
        
        training_status[ticker] = "COMPLETED"
        print(f" Dynamic training for {ticker} complete!")
        
    except Exception as e:
        training_status[ticker] = f"FAILED: {str(e)}"
        print(f" Dynamic training for {ticker} failed: {e}")

@dynamic_learn_bp.route('/add-stock', methods=['POST'])
def add_stock():
    data = request.json
    ticker = data.get('ticker')
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
        
    if ticker in training_status and training_status[ticker] not in ["COMPLETED", "FAILED"]:
        return jsonify({"status": training_status[ticker], "message": "Training already in progress"}), 200

    # Start training in background
    thread = threading.Thread(target=train_new_ticker_task, args=(ticker,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "STARTED",
        "ticker": ticker,
        "message": f"Training initiated for {ticker}. Check status via /api/train/status/{ticker}"
    }), 202

@dynamic_learn_bp.route('/status/<ticker>', methods=['GET'])
def get_status(ticker):
    status = training_status.get(ticker, "NOT_FOUND")
    return jsonify({"ticker": ticker, "status": status})
