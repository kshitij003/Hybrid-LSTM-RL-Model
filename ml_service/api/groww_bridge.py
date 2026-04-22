from flask import Blueprint, request, jsonify
import os
import logging
import pyotp
try:
    from growwapi import GrowwAPI
except ImportError:
    GrowwAPI = None

trade_bp = Blueprint('trade', __name__)

# Configure logging
logger = logging.getLogger("GrowwBridge")

# Configuration
GROWW_API_KEY = os.getenv("GROWW_API_KEY")
GROWW_API_SECRET = os.getenv("GROWW_API_SECRET")
GROWW_TOTP_SECRET = os.getenv("GROWW_TOTP_SECRET")

# Mock Mode (Default to True for safety)
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

def get_groww_client():
    """
    Initializes and returns a GrowwAPI client using API Key and Secret.
    """
    if GrowwAPI is None:
        raise ImportError("growwapi library not installed. Run 'pip install growwapi'")

    if not all([GROWW_API_KEY, GROWW_API_SECRET]):
        logger.error("Missing GROWW_API_KEY or GROWW_API_SECRET in .env")
        return None

    try:
        # Get Access Token using API Key and Secret (as per sample)
        access_token = GrowwAPI.get_access_token(
            api_key=GROWW_API_KEY, 
            secret=GROWW_API_SECRET
        )

        return GrowwAPI(access_token)
    except Exception as e:
        logger.error(f"Failed to authenticate with Groww: {str(e)}")
        return None

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
        groww = get_groww_client()
        if not groww:
            return False, "Authentication Failed"

        logger.info(f"🚀 Executing REAL {trade_type} for {ticker}...")
        
        # Strip .NS for Groww
        symbol = ticker.split('.')[0]
        
        # Place order using parameters from the official sample
        order = groww.place_order(
            trading_symbol=symbol, 
            quantity=quantity, 
            validity=groww.VALIDITY_DAY,
            exchange=groww.EXCHANGE_NSE, 
            segment=groww.SEGMENT_CASH,
            product=groww.PRODUCT_MIS,
            order_type=groww.ORDER_TYPE_MARKET,
            transaction_type=groww.TRANSACTION_TYPE_BUY if trade_type.upper() == "BUY" else groww.TRANSACTION_TYPE_SELL
        )
        
        logger.info(f"✅ Order placed successfully: {order}")
        return True, str(order)
        
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
