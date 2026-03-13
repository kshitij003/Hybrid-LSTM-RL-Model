"""
Standalone ML Accuracy Evaluator
Tests models directly without needing Spring Boot
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import torch
from models.lstm_model import LSTMPredictor
from stable_baselines3 import PPO
from models.multi_stock_env import MultiStockPortfolioEnv
import requests
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
FLASK_URL = "http://localhost:8000"


def test_sentiment_service():
    """Test FinBERT sentiment analysis"""
    print(f"\n{'='*70}")
    print(f"🎭 TESTING SENTIMENT ANALYSIS")
    print(f"{'='*70}\n")
    
    test_cases = [
        {"text": "Apple stock soars to record highs on strong earnings beat", "expected": "positive"},
        {"text": "Tesla faces major production delays and massive recalls", "expected": "negative"},
        {"text": "Microsoft maintains steady market position this quarter", "expected": "neutral"},
        {"text": "Amazon crushes expectations with record holiday sales", "expected": "positive"},
        {"text": "Google stock plummets on regulatory concerns", "expected": "negative"},
        {"text": "NVIDIA reports quarterly results in line with estimates", "expected": "neutral"}
    ]
    
    correct = 0
    results = []
    
    for case in test_cases:
        try:
            response = requests.post(
                f"{FLASK_URL}/api/news/analyze",
                json={"text": case['text']},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result['sentiment']['label']
                score = result['sentiment']['score']
                confidence = result['sentiment']['confidence']
                
                is_correct = predicted == case['expected']
                if is_correct:
                    correct += 1
                
                results.append({
                    'text': case['text'][:50] + "...",
                    'expected': case['expected'],
                    'predicted': predicted,
                    'score': score,
                    'confidence': confidence,
                    'correct': '✅' if is_correct else '❌'
                })
                
                print(f"   {results[-1]['correct']} Expected: {case['expected']:8} | Got: {predicted:8} | Score: {score:+.2f} | Confidence: {confidence:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing text: {e}")
    
    accuracy = (correct / len(test_cases)) * 100
    
    print(f"\n   📊 FinBERT Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(test_cases),
        'results': results
    }


def test_inference_speed(iterations=20):
    """Test ML inference speed"""
    print(f"\n{'='*70}")
    print(f"⚡ TESTING INFERENCE SPEED")
    print(f"{'='*70}\n")
    
    latencies = []
    
    # Prepare test payload with correct format
    payload = {
        "currentCash": 5000.0,
        "currentHoldings": {
            "AAPL": 1000.0,
            "MSFT": 1000.0,
            "GOOGL": 1000.0,
            "AMZN": 1000.0,
            "TSLA": 1000.0
        },
        "marketData": {
            stock: [
                {"date": f"2024-{i:02d}-01", "close": 150.0 + i, "volume": 50000000, "sentimentScore": 0.5}
                for i in range(1, 61)
            ]
            for stock in STOCKS
        }
    }
    
    print(f"   Running {iterations} inference tests...")
    
    for i in range(iterations):
        try:
            start = datetime.now()
            response = requests.post(
                f"{FLASK_URL}/api/predict",
                json=payload,
                timeout=5
            )
            end = datetime.now()
            
            if response.status_code == 200:
                latency_ms = (end - start).total_seconds() * 1000
                latencies.append(latency_ms)
                
                if (i + 1) % 5 == 0:
                    print(f"   Progress: {i+1}/{iterations} | Avg: {np.mean(latencies):.0f}ms")
        
        except Exception as e:
            print(f"   ❌ Request {i+1} failed: {e}")
    
    if latencies:
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\n   ⚡ Speed Metrics:")
        print(f"      Average: {avg_latency:.1f} ms")
        print(f"      P50: {p50_latency:.1f} ms")
        print(f"      P95: {p95_latency:.1f} ms")
        print(f"      P99: {p99_latency:.1f} ms")
        
        return {
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'successful_requests': len(latencies),
            'total_requests': iterations
        }
    else:
        print(f"\n   ❌ No successful requests")
        return None


def test_model_availability():
    """Check which models are available"""
    print(f"\n{'='*70}")
    print(f"📦 CHECKING MODEL AVAILABILITY")
    print(f"{'='*70}\n")
    
    try:
        response = requests.get(f"{FLASK_URL}/api/models", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            
            print(f"   Found {len(models)} model(s):")
            for model in models:
                status = "🟢 ACTIVE" if model.get('is_active') else "⚪ Inactive"
                print(f"      {status} {model.get('model_id')}")
                print(f"         Created: {model.get('created_at', 'Unknown')}")
            
            return models
        else:
            print(f"   ❌ Failed to get models: {response.status_code}")
            return []
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []


def test_news_fetching():
    """Test news fetching capability"""
    print(f"\n{'='*70}")
    print(f"📰 TESTING NEWS FETCHING")
    print(f"{'='*70}\n")
    
    test_stock = "AAPL"
    
    try:
        response = requests.post(
            f"{FLASK_URL}/api/news/fetch",
            json={"ticker": test_stock, "days": 3},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            avg_sentiment = data.get('averageSentiment', 0)
            
            print(f"   ✅ Fetched {len(articles)} articles for {test_stock}")
            print(f"   📊 Average Sentiment: {avg_sentiment:+.3f}")
            
            if articles:
                print(f"\n   Sample Article:")
                print(f"      Title: {articles[0].get('title', 'N/A')[:70]}...")
                print(f"      Sentiment: {articles[0]['sentiment']['label']} ({articles[0]['sentiment']['score']:+.2f})")
            
            return {
                'articles_fetched': len(articles),
                'average_sentiment': avg_sentiment,
                'test_stock': test_stock
            }
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_training_status():
    """Check training jobs"""
    print(f"\n{'='*70}")
    print(f"🤖 CHECKING TRAINING STATUS")
    print(f"{'='*70}\n")
    
    try:
        response = requests.get(f"{FLASK_URL}/api/train/list", timeout=5)
        
        if response.status_code == 200:
            jobs = response.json().get('jobs', [])
            
            if jobs:
                print(f"   Found {len(jobs)} training job(s):")
                for job in jobs[:5]:  # Show last 5
                    status_icon = {
                        'COMPLETED': '✅',
                        'FAILED': '❌',
                        'IN_PROGRESS': '🔄',
                        'QUEUED': '⏳'
                    }.get(job.get('status'), '❓')
                    
                    print(f"      {status_icon} {job.get('id')} - {job.get('status')}")
                    if job.get('config'):
                        print(f"         Stocks: {job['config'].get('stocks', [])}")
            else:
                print(f"   No training jobs found")
            
            return jobs
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return []
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []


def run_comprehensive_test():
    """Run all standalone tests"""
    print(f"\n{'='*70}")
    print(f"🚀 ML SERVICE STANDALONE EVALUATION")
    print(f"{'='*70}")
    print(f"Testing Flask ML service at: {FLASK_URL}")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Test 1: Model availability
    print("\n[1/6] Checking available models...")
    results['models'] = test_model_availability()
    
    # Test 2: Sentiment analysis
    print("\n[2/6] Testing sentiment analysis...")
    results['sentiment'] = test_sentiment_service()
    
    # Test 3: News fetching
    print("\n[3/6] Testing news fetching...")
    results['news'] = test_news_fetching()
    
    # Test 4: Inference speed
    print("\n[4/6] Testing inference speed...")
    results['speed'] = test_inference_speed(iterations=10)
    
    # Test 5: Training status
    print("\n[5/6] Checking training history...")
    results['training'] = test_training_status()
    
    # Generate summary
    print(f"\n{'='*70}")
    print(f"📋 EVALUATION SUMMARY")
    print(f"{'='*70}\n")
    
    if results['sentiment']:
        acc = results['sentiment']['accuracy']
        emoji = '✅' if acc >= 80 else '⚠️' if acc >= 60 else '❌'
        print(f"{emoji} Sentiment Accuracy: {acc:.1f}%")
    
    if results['speed']:
        latency = results['speed']['avg_latency_ms']
        emoji = '✅' if latency < 500 else '⚠️' if latency < 1000 else '❌'
        print(f"{emoji} Inference Latency: {latency:.0f} ms")
    
    if results['news']:
        count = results['news']['articles_fetched']
        emoji = '✅' if count > 0 else '❌'
        print(f"{emoji} News Fetching: {count} articles")
    
    if results['models']:
        active = sum(1 for m in results['models'] if m.get('is_active'))
        emoji = '✅' if active > 0 else '⚠️'
        print(f"{emoji} Active Models: {active}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"standalone_eval_{timestamp}.json"
    
    with open(filename, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'timestamp': timestamp,
            'sentiment': results.get('sentiment'),
            'speed': results.get('speed'),
            'news': results.get('news'),
            'models_count': len(results.get('models', [])),
            'training_jobs_count': len(results.get('training', []))
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Report saved to: {filename}")
    print(f"\n{'='*70}\n")
    
    return results


if __name__ == "__main__":
    run_comprehensive_test()
