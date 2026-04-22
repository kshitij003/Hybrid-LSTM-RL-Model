"""
News Fetching and Sentiment Analysis Service
Fetches real stock news and analyzes sentiment using FinBERT
"""

from flask import Blueprint, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from datetime import datetime, timedelta
import os

news_bp = Blueprint('news', __name__)

# Load FinBERT model for financial sentiment analysis
print("📰 Loading FinBERT sentiment analysis model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()
print("✅ FinBERT model loaded successfully!")

# News API configuration (using NewsAPI.org - free tier)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'YOUR_API_KEY_HERE')  # Get from newsapi.org
NEWS_API_URL = "https://newsapi.org/v2/everything"


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of text using FinBERT
    Returns: {
        'score': float (-1 to 1),
        'label': 'positive' | 'neutral' | 'negative',
        'confidence': float (0 to 1)
    }
    """
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT outputs: [positive, negative, neutral]
        probabilities = predictions[0].tolist()
        labels = ['positive', 'negative', 'neutral']
        
        # Get dominant sentiment
        max_idx = probabilities.index(max(probabilities))
        sentiment_label = labels[max_idx]
        confidence = probabilities[max_idx]
        
        # Convert to score: positive=1, neutral=0, negative=-1
        score_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
        # Weighted score
        sentiment_score = (
            probabilities[0] * 1.0 +   # positive
            probabilities[1] * -1.0 +  # negative  
            probabilities[2] * 0.0     # neutral
        )
        
        return {
            'score': round(sentiment_score, 4),
            'label': sentiment_label,
            'confidence': round(confidence, 4),
            'probabilities': {
                'positive': round(probabilities[0], 4),
                'negative': round(probabilities[1], 4),
                'neutral': round(probabilities[2], 4)
            }
        }
    
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        return {
            'score': 0.0,
            'label': 'neutral',
            'confidence': 0.0,
            'error': str(e)
        }


def fetch_news(ticker: str, days: int = 7) -> list:
    """
    Fetch news articles for a stock ticker and score them with FinBERT.
    Args:
        ticker: Stock symbol e.g. 'RELIANCE.NS' or 'AAPL'
        days: Number of days to look back
    Returns:
        List of news articles with FinBERT sentiment scores
    """
    try:
        # ── Company name map (used as search query) ───────────────────────────
        # Indian NSE stocks (strip .NS suffix for lookup)
        company_map = {
            # ── Indian Nifty 50 ──────────────────────────────────────────────
            'RELIANCE':    'Reliance Industries',
            'TCS':         'Tata Consultancy Services TCS',
            'HDFCBANK':    'HDFC Bank',
            'INFY':        'Infosys',
            'ICICIBANK':   'ICICI Bank',
            'HINDUNILVR':  'Hindustan Unilever HUL',
            'SBIN':        'State Bank of India SBI',
            'BAJFINANCE':  'Bajaj Finance',
            'BHARTIARTL':  'Bharti Airtel',
            'KOTAKBANK':   'Kotak Mahindra Bank',
            'WIPRO':       'Wipro',
            'HCLTECH':     'HCL Technologies',
            'ASIANPAINT':  'Asian Paints',
            'AXISBANK':    'Axis Bank',
            'MARUTI':      'Maruti Suzuki',
            'SUNPHARMA':   'Sun Pharmaceutical',
            'TITAN':       'Titan Company',
            'TATAMOTORS':  'Tata Motors',
            'TATASTEEL':   'Tata Steel',
            'ULTRACEMCO':  'UltraTech Cement',
            'ADANIENT':    'Adani Enterprises',
            'ADANIPORTS':  'Adani Ports',
            'POWERGRID':   'Power Grid Corporation',
            'NTPC':        'NTPC Limited',
            'ONGC':        'ONGC Oil Natural Gas',
            'COALINDIA':   'Coal India',
            'GRASIM':      'Grasim Industries',
            'TECHM':       'Tech Mahindra',
            'LTIM':        'LTIMindtree',
            'JSWSTEEL':    'JSW Steel',
            # ── US Stocks ────────────────────────────────────────────────────
            'AAPL':  'Apple',
            'MSFT':  'Microsoft',
            'GOOGL': 'Google Alphabet',
            'AMZN':  'Amazon',
            'TSLA':  'Tesla',
            'NVDA':  'Nvidia',
            'META':  'Meta Facebook',
        }

        # Strip exchange suffix (.NS / .BO) for map lookup
        base_ticker = ticker.split('.')[0].upper()
        company_name = company_map.get(base_ticker, base_ticker)

        # Add "stock NSE" suffix for Indian tickers to narrow results
        if ticker.endswith('.NS') or ticker.endswith('.BO'):
            query = f"{company_name} stock NSE India"
        else:
            query = f"{company_name} stock"

        # Calculate date range
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Fetch from NewsAPI
        params = {
            'q':        query,
            'from':     start_date.strftime('%Y-%m-%d'),
            'to':       end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy':   'publishedAt',
            'apiKey':   NEWS_API_KEY,
            'pageSize': 20
        }

        print(f"📰 Fetching news for {ticker} (query: '{query}')...")
        response = requests.get(NEWS_API_URL, params=params, timeout=10)

        if response.status_code != 200:
            print(f"⚠️  NewsAPI error: {response.status_code}")
            return []

        articles = response.json().get('articles', [])
        print(f"   Found {len(articles)} articles")

        # FinBERT-score each article
        news_with_sentiment = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".strip()
            if not text:
                continue
            sentiment = analyze_sentiment(text)
            news_with_sentiment.append({
                'title':       article.get('title'),
                'description': article.get('description'),
                'content':     article.get('content'),
                'source':      article.get('source', {}).get('name'),
                'publishedAt': article.get('publishedAt'),
                'url':         article.get('url'),
                'sentiment':   sentiment,
            })

        print(f"   ✅ Scored {len(news_with_sentiment)} articles with FinBERT")
        return news_with_sentiment

    except Exception as e:
        print(f"❌ News fetch failed for {ticker}: {e}")
        return []


@news_bp.route('/fetch', methods=['POST'])
def fetch_news_endpoint():
    """
    Endpoint to fetch and analyze news
    POST /api/news/fetch
    Body: {
        "ticker": "AAPL",
        "days": 7
    }
    """
    try:
        data = request.json
        ticker = data.get('ticker')
        days = data.get('days', 7)
        
        if not ticker:
            return jsonify({'error': 'ticker is required'}), 400
        
        news = fetch_news(ticker, days)
        
        # Calculate average sentiment
        if news:
            avg_sentiment = sum(article['sentiment']['score'] for article in news) / len(news)
        else:
            avg_sentiment = 0.0
        
        return jsonify({
            'ticker': ticker,
            'articles': news,
            'totalArticles': len(news),
            'averageSentiment': round(avg_sentiment, 4),
            'dateRange': {
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d')
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@news_bp.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Endpoint to analyze sentiment of any text
    POST /api/news/analyze
    Body: {
        "text": "Apple stock soars to new highs!"
    }
    """
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'text is required'}), 400
        
        sentiment = analyze_sentiment(text)
        
        return jsonify({
            'text': text,
            'sentiment': sentiment
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@news_bp.route('/batch-fetch', methods=['POST'])
def batch_fetch_news():
    """
    Fetch news for multiple stocks
    POST /api/news/batch-fetch
    Body: {
        "stocks": ["AAPL", "MSFT", "GOOGL"],
        "days": 7
    }
    """
    try:
        data = request.json
        stocks = data.get('stocks', [])
        days = data.get('days', 7)
        
        if not stocks:
            return jsonify({'error': 'stocks list is required'}), 400
        
        results = {}
        for ticker in stocks:
            news = fetch_news(ticker, days)
            
            if news:
                avg_sentiment = sum(article['sentiment']['score'] for article in news) / len(news)
            else:
                avg_sentiment = 0.0
            
            results[ticker] = {
                'articles': news,
                'totalArticles': len(news),
                'averageSentiment': round(avg_sentiment, 4)
            }
        
        return jsonify({
            'stocks': results,
            'dateRange': {
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d')
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
