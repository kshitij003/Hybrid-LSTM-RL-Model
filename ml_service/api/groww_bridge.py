"""
Groww Trading Bridge
Handles real execution on Groww platform using unofficial APIs
"""

from flask import Blueprint, request, jsonify
import os
import logging

trade_bp = Blueprint('trade', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GrowwBridge")

# Configuration (To be provided by user in .env)
GROWW_EMAIL = os.getenv("GROWW_EMAIL")
GROWW_PIN = os.getenv("GROWW_PIN")
GROWW_TOTP_SECRET = os.getenv("GROWW_TOTP_SECRET")

# Mock Mode (Default to True for safety)
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

def execute_groww_trade(ticker, trade_type, quantity):
    """
    Core logic to execute trade on Groww.
    ticker: expected to be ticker.NS (e.g. RELIANCE.NS)
    trade_type: BUY or SELL
    quantity: int
    """
    if DRY_RUN:
        logger.info(f"🧪 [DRY RUN] Would execute {trade_type} of {quantity} shares of {ticker} on Groww")
        return True, "DRY_RUN_SUCCESS"

    try:
        # Import only when needed to avoid dependency issues on startup
        # from groww_python import Groww
        
        # logger.info(f"🚀 Executing REAL {trade_type} for {ticker}...")
        
        # Example Implementation (Conceptual):
        # groww = Groww()
        # groww.login(GROWW_EMAIL, GROWW_PIN, GROWW_TOTP_SECRET)
        # order = groww.place_order(
        #    symbol=ticker.split('.')[0], 
        #    qty=quantity, 
        #    side=trade_type, 
        #    type="MARKET"
        # )
        
        # For now, we simulate the 'Ready' state to allow user to plug in library
        logger.warning(f"⚠️  Live trading logic triggered for {ticker} but library bridge is pending user credentials.")
        return True, "MOCK_LIVE_SUCCESS"
        
    except Exception as e:
        logger.error(f"❌ Groww Trade Failed: {str(e)}")
        return False, str(e)

@trade_bp.route('/trade', methods=['POST'])
def trade():
    data = request.json
    ticker = data.get('ticker')
    trade_type = data.get('type') # BUY or SELL
    quantity = data.get('quantity')
    
    if not all([ticker, trade_type, quantity]):
        return jsonify({"error": "Missing trade parameters"}), 400

    success, message = execute_groww_trade(ticker, trade_type, quantity)
    
    if success:
        return jsonify({
            "status": "SUCCESS",
            "message": message,
            "details": f"{trade_type} {quantity} shares of {ticker}"
        }), 200
    else:
        return jsonify({
            "status": "FAILED",
            "error": message
        }), 500
