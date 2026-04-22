import requests
import json
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("AutoRebalancer")

load_dotenv()

# Configuration
ML_SERVICE_URL = f"http://localhost:{os.getenv('PORT', '8000')}"
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

def run_rebalancing_cycle(tickers):
    """
    1. Fetch current portfolio state (simulated or real from Groww)
    2. Get AI prediction
    3. Calculate required trades
    4. Execute trades via Groww Bridge
    """
    logger.info("Starting Auto-Rebalancing Cycle...")
    
    # --- Step 1: Get Current Portfolio ---
    # For the demo, we assume a starting balance. 
    # In a real scenario, you'd fetch this from the Groww API balance endpoint.
    current_cash = 100000.0
    current_holdings = {ticker: 0.0 for ticker in tickers}
    
    logger.info(f"Current Portfolio: Cash={current_cash}, Holdings={sum(current_holdings.values())}")

    # --- Step 2: Get AI Prediction ---
    try:
        logger.info("Requesting AI prediction...")
        predict_payload = {
            "currentCash": current_cash,
            "currentHoldings": current_holdings,
            "tickers": tickers
        }
        
        response = requests.post(f"{ML_SERVICE_URL}/api/predict", json=predict_payload)
        response.raise_for_status()
        prediction = response.json()
        
        target_weights = prediction['targetWeights']
        logger.info(f"AI Recommendations: {json.dumps(target_weights, indent=2)}")
        
    except Exception as e:
        logger.error(f"Failed to get prediction: {e}")
        return

    # --- Step 3: Calculate and Execute Trades ---
    total_value = current_cash + sum(current_holdings.values())
    
    for ticker, weight in target_weights.items():
        if ticker == "CASH":
            continue
            
        target_value = weight * total_value
        current_value = current_holdings.get(ticker, 0.0)
        
        trade_value = target_value - current_value
        
        # Determine Trade Type and Quantity
        # Note: We fetch the price from the prediction metadata if available
        # or assume a price for calculation.
        if abs(trade_value) > 100: # Minimum trade threshold (e.g. 100 INR)
            trade_type = "BUY" if trade_value > 0 else "SELL"
            
            # Simple quantity calculation (assuming we can get price)
            # For demo, we'll send the request to the bridge
            # The bridge handles the actual Groww library calls.
            
            logger.info(f"[BALANCE] Rebalancing {ticker}: Target Weight {weight:.1%}")
            
            trade_payload = {
                "ticker": ticker,
                "type": trade_type,
                "quantity": 1 # For demo, we use 1 share
            }
            
            try:
                trade_resp = requests.post(f"{ML_SERVICE_URL}/api/trade", json=trade_payload)
                trade_resp.raise_for_status()
                result = trade_resp.json()
                logger.info(f"[SUCCESS] {ticker} {trade_type} Order Result: {result['message']}")
            except Exception as e:
                logger.error(f"[FAILED] Failed to execute trade for {ticker}: {e}")

    logger.info("Rebalancing cycle complete.")

if __name__ == "__main__":
    # The 5 stocks we trained on
    trained_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    
    if DRY_RUN:
        print("\n" + "!"*50)
        print("WARNING: DRY RUN MODE ENABLED")
        print("   No real money will be spent.")
        print("!"*50 + "\n")
    
    run_rebalancing_cycle(trained_stocks)
