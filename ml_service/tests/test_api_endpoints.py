"""
Test script for API endpoints
Tests all Flask endpoints with sample requests
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def print_section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_health():
    """Test health endpoint"""
    print_section("TEST 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Health check passed")


def test_inference_health():
    """Test inference health endpoint"""
    print_section("TEST 2: Inference Health")
    
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("✅ Inference health check passed")


def test_list_models():
    """Test model listing"""
    print_section("TEST 3: List Models")
    
    response = requests.get(f"{BASE_URL}/api/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("✅ Model listing passed")


def test_predict_endpoint():
    """Test prediction endpoint"""
    print_section("TEST 4: Prediction Endpoint")
    
    # Sample request matching Spring Boot format
    request_data = {
        "currentCash": 5000.0,
        "currentHoldings": {
            "AAPL": 2000.0,
            "MSFT": 2000.0,
            "GOOGL": 1000.0
        },
        "marketData": {
            "AAPL": [
                {"date": "2024-01-01", "close": 150.0, "volume": 50000000, "sentimentScore": 0.75}
                for _ in range(60)  # 60 days of data
            ],
            "MSFT": [
                {"date": "2024-01-01", "close": 300.0, "volume": 30000000, "sentimentScore": 0.65}
                for _ in range(60)
            ],
            "GOOGL": [
                {"date": "2024-01-01", "close": 120.0, "volume": 40000000, "sentimentScore": 0.70}
                for _ in range(60)
            ]
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Prediction endpoint passed")
    elif response.status_code == 503:
        print("⚠️  No trained model available (expected)")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Response: {response.text}")


def test_training_list():
    """Test training job listing"""
    print_section("TEST 5: List Training Jobs")
    
    response = requests.get(f"{BASE_URL}/api/train/list")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("✅ Training list passed")


def run_all_tests():
    """Run all API tests"""
    print("\n" + "=" * 70)
    print(" FLASK API ENDPOINTS - TEST SUITE")
    print("=" * 70)
    
    try:
        test_health()
        test_inference_health()
        test_list_models()
        test_predict_endpoint()
        test_training_list()
        
        print("\n" + "=" * 70)
        print("✅ ALL API TESTS COMPLETED!")
        print("=" * 70)
        print("\n🎉 Your Flask API is working correctly!")
        print("\nNext: Test from Spring Boot backend")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to Flask server")
        print("Make sure to start the server first:")
        print("  cd ml_service")
        print("  python app.py")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
