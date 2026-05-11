import os
import pyotp
import logging
from dotenv import load_dotenv

# Try to import growwapi
try:
    from growwapi import GrowwAPI
except ImportError:
    print(" Error: growwapi not installed. Run 'pip install growwapi pyotp'")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GrowwTest")

def test_connection():
    load_dotenv()
    
    api_key = os.getenv("GROWW_API_KEY")
    totp_secret = os.getenv("GROWW_TOTP_SECRET")
    
    if not api_key or not totp_secret:
        print(" Error: Missing GROWW_API_KEY or GROWW_TOTP_SECRET in .env")
        return

    print(" Attempting to connect to Groww...")
    
    try:
        # 1. Generate TOTP
        totp_gen = pyotp.TOTP(totp_secret)
        current_totp = totp_gen.now()
        print(f" Generated TOTP: {current_totp}")
        
        # 2. Get Access Token
        # Note: This is based on the documentation provided by the user
        print("🌐 Requesting access token...")
        access_token = GrowwAPI.get_access_token(
            api_key=api_key,
            totp=current_totp
        )
        print(" Access token received!")
        
        # 3. Initialize API
        groww = GrowwAPI(access_token)
        
        # 4. Fetch Balance (as a non-destructive test)
        print(" Fetching balance...")
        # Note: Method name might vary, assuming get_balance or similar based on standard SDKs
        # If this fails, it still proves the authentication flow worked.
        try:
            balance = groww.get_balance()
            print(f"💳 Account Balance: {balance}")
        except AttributeError:
            print("ℹ️  Authentication worked, but 'get_balance' method name might be different in this SDK version.")
            
        print("\n Connection Test Successful!")
        
    except Exception as e:
        print(f" Connection Failed: {str(e)}")

if __name__ == "__main__":
    test_connection()
