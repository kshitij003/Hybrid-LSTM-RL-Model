from flask import Blueprint, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

news_bp = Blueprint('news', __name__)

# Load FinBERT model for sentiment analysis
# ProsusAI/finbert is the standard for financial sentiment
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes sentiment of text using FinBERT.
    Returns label and score.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT labels: 0 -> positive, 1 -> negative, 2 -> neutral
        probabilities = predictions[0].tolist()
        labels = ["positive", "negative", "neutral"]
        max_idx = np.argmax(probabilities)
        
        # Calculate a single score: positive - negative
        sentiment_score = probabilities[0] - probabilities[1]
        
        return {
            "label": labels[max_idx],
            "score": sentiment_score,
            "probabilities": {
                "positive": probabilities[0],
                "negative": probabilities[1],
                "neutral": probabilities[2]
            }
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {"label": "neutral", "score": 0.0, "probabilities": {}}

import numpy as np

def fetch_news_finnhub(ticker: str, days: int = 7) -> list:
    """
    Fetch news articles from Finnhub.
    """
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == 'YOUR_FINNHUB_KEY_HERE':
        return []

    try:
        # Use full ticker for Indian stocks (e.g. RELIANCE.NS)
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            'symbol': ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': FINNHUB_API_KEY
        }

        print(f" Fetching Finnhub news for {ticker}...")
        response = requests.get(FINNHUB_URL, params=params, timeout=10)

        if response.status_code != 200:
            print(f"  Finnhub error: {response.status_code}")
            return []

        articles = response.json()
        if not isinstance(articles, list):
            return []
            
        print(f"   Found {len(articles)} articles on Finnhub")

        news_with_sentiment = []
        for article in articles:
            headline = article.get('headline', '')
            summary = article.get('summary', '')
            text = f"{headline} {summary}".strip()
            
            if not text: continue
                
            sentiment = analyze_sentiment(text)
            pub_date = datetime.fromtimestamp(article.get('datetime', 0))
            
            news_with_sentiment.append({
                'title':       headline,
                'description': summary,
                'content':     summary,
                'source':      article.get('source', 'Finnhub'),
                'publishedAt': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                'url':         article.get('url'),
                'sentiment':   sentiment,
            })
        return news_with_sentiment
    except Exception as e:
        print(f" Finnhub fetch failed for {ticker}: {e}")
        return []

def fetch_news_yahoo(ticker: str) -> list:
    """
    Fetch news articles from Yahoo Finance.
    """
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=20"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        print(f" Fetching Yahoo Finance news for {ticker}...")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return []
            
        data = response.json()
        news_items = data.get('news', [])
        print(f"   Found {len(news_items)} articles on Yahoo Finance")

        news_with_sentiment = []
        for item in news_items:
            title = item.get('title', '')
            if not title: continue
            
            sentiment = analyze_sentiment(title)
            print(f"      - '{title[:50]}...' | Score: {sentiment['score']:+.4f}")
            
            ts = item.get('providerPublishTime', 0)
            pub_date = datetime.fromtimestamp(ts) if ts > 0 else datetime.now()
            
            news_with_sentiment.append({
                'title':       title,
                'description': item.get('publisher', ''),
                'content':     title,
                'source':      item.get('publisher', 'Yahoo Finance'),
                'publishedAt': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                'url':         item.get('link'),
                'sentiment':   sentiment,
            })
        return news_with_sentiment
    except Exception as e:
        print(f" Yahoo news fetch failed: {e}")
        return []

def fetch_news_google(ticker: str) -> list:
    """
    Fetch news articles from Google News RSS.
    """
    try:
        import xml.etree.ElementTree as ET
        search_query = f"{ticker.split('.')[0]} stock news"
        url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        print(f" Fetching Google News for {ticker}...")
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return []
            
        root = ET.fromstring(response.content)
        news_with_sentiment = []
        
        for item in root.findall('.//item')[:15]:
            title = item.find('title').text
            pub_date_str = item.find('pubDate').text
            
            try:
                pub_date = datetime.strptime(pub_date_str[:25], '%a, %d %b %Y %H:%M:%S')
            except:
                pub_date = datetime.now()

            sentiment = analyze_sentiment(title)
            
            news_with_sentiment.append({
                'title':       title,
                'description': title,
                'content':     title,
                'source':      'Google News',
                'publishedAt': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                'url':         item.find('link').text,
                'sentiment':   sentiment,
            })
        print(f"   Found {len(news_with_sentiment)} articles on Google News")
        return news_with_sentiment
    except Exception as e:
        print(f" Google News fetch failed: {e}")
        return []

def fetch_news(ticker: str, days: int = 7) -> list:
    """
    Fetch news articles for a stock ticker.
    Order: Google News -> Yahoo Finance -> Finnhub
    """
    news = fetch_news_google(ticker)
    if news: return news

    news = fetch_news_yahoo(ticker)
    if news: return news
    
    news = fetch_news_finnhub(ticker, days)
    if news: return news
    
    return []

@news_bp.route('/sentiment', methods=['GET'])
def get_sentiment():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    
    news = fetch_news(ticker)
    if not news:
        return jsonify({"ticker": ticker, "sentiment_score": 0.0, "news_count": 0})
    
    avg_score = np.mean([n['sentiment']['score'] for n in news])
    return jsonify({
        "ticker": ticker,
        "sentiment_score": avg_score,
        "news_count": len(news),
        "latest_headlines": [n['title'] for n in news[:3]]
    })

@news_bp.route('/headlines', methods=['GET'])
def get_headlines():
    """Fetch and score headlines for all user portfolio stocks."""
    # Try to load stocks from portfolio preferences if available
    try:
        from api.portfolio_preferences import _load_preferences
        prefs = _load_preferences()
        stocks = prefs.get('stocks', ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"])
    except:
        stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    
    all_news = []
    # Limit to first 3 stocks to prevent timeout/rate limits
    for ticker in stocks[:3]:
        try:
            articles = fetch_news(ticker, days=3)
            for a in articles:
                all_news.append({
                    "headline": a['title'],
                    "source": a['source'],
                    "sentiment": a['sentiment']['label'].upper(),
                    "score": a['sentiment']['score'],
                    "ticker": ticker,
                    "publishedAt": a.get('publishedAt', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                })
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            
    # Sort by publishedAt descending
    all_news.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
    return jsonify({"headlines": all_news[:20], "total": len(all_news)})
