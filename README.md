# 🤖 Hybrid LSTM+RL Trading System

AI-powered portfolio management system combining **LSTM time-series prediction** with **PPO reinforcement learning** for automated trading decisions.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Java](https://img.shields.io/badge/Java-17+-orange)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.0-green)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue)

---

## 🌟 Features

### **AI & Machine Learning**
- ✅ Multi-stock LSTM time-series forecasting
- ✅ PPO reinforcement learning for portfolio optimization
- ✅ FinBERT financial sentiment analysis
- ✅ Automated quarterly model retraining
- ✅ Real-time inference API

### **Data Pipeline**
- ✅ Yahoo Finance integration
- ✅ NewsAPI real-time news fetching
- ✅ PostgreSQL as single source of truth
- ✅ Automated data compression & maintenance
- ✅ Real sentiment scoring (not dummy values!)

### **Portfolio Management**
- ✅ AI-powered portfolio rebalancing
- ✅ Risk validation & circuit breakers
- ✅ Transaction fee calculation (0.1%)
- ✅ Complete audit trail
- ✅ NAV calculation

### **Automation**
- ✅ Daily news sync (midnight)
- ✅ Monthly data compression (1st of month)
- ✅ Quarterly model retraining (Jan/Apr/Jul/Oct)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Spring Boot Backend             │
│         (Port 8080)                     │
│                                         │
│  - Portfolio Management                │
│  - Trade Execution                     │
│  - Data Sync & Export                  │
│  - Scheduled Tasks                     │
└──────────────┬──────────────────────────┘
               │ HTTP REST
               ↓
┌─────────────────────────────────────────┐
│         Flask ML Service                │
│         (Port 8000)                     │
│                                         │
│  - LSTM Models                         │
│  - PPO Agent                           │
│  - FinBERT Sentiment                   │
│  - Training Pipeline                   │
└─────────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│         PostgreSQL Database             │
│         (Port 5432)                     │
│                                         │
│  - Stock Prices                        │
│  - News & Sentiment                    │
│  - Portfolios & Transactions           │
│  - Model Signals                       │
└─────────────────────────────────────────┘
```

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Spring Boot 3.0, Java 17 |
| **ML Service** | Flask 2.3, Python 3.9+ |
| **Database** | PostgreSQL 13+ |
| **ML Frameworks** | PyTorch, Stable-Baselines3, Transformers |
| **Data Sources** | Yahoo Finance, NewsAPI |
| **Models** | LSTM (time-series), PPO (RL), FinBERT (sentiment) |

---

## 🚀 Quick Start

### **Prerequisites**
- Java 17+
- Python 3.9+
- PostgreSQL 13+
- NewsAPI key (free from [newsapi.org](https://newsapi.org/register))

### **1. Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/Hybrid-LSTM-RL-Project.git
cd Hybrid-LSTM-RL-Project
```

### **2. Setup Database**
```bash
# Create PostgreSQL database
psql -U postgres
CREATE DATABASE trading_db;
```

### **3. Setup ML Service**
```bash
cd ml_service

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your NEWS_API_KEY

# Start Flask server
python app.py
```

### **4. Setup Spring Boot Backend**
```bash
cd trading_backend/trading_backend

# Configure application.properties
# Update database credentials

# Run Spring Boot
./mvnw spring-boot:run
```

### **5. Initialize Data**
```bash
# Sync initial price data
curl -X POST http://localhost:8080/api/portfolio/sync-data

# Fetch news with sentiment
curl -X POST http://localhost:8080/api/news/sync-batch \
  -H "Content-Type: application/json" \
  -d '{"stocks": ["AAPL","MSFT","GOOGL","AMZN","TSLA"], "days": 7}'
```

### **6. Train Model (Optional - pre-trained available)**
```bash
curl -X POST "http://localhost:8080/api/ml/trigger-training?daysOfData=90"
```

---

## 📊 API Endpoints

### **Spring Boot (Port 8080)**

#### Portfolio Management
- `GET /api/portfolio/{id}` - Get portfolio details
- `GET /api/portfolio/{id}/nav` - Calculate net asset value
- `POST /api/portfolio/{id}/rebalance` - AI-powered rebalancing

#### ML Training
- `POST /api/ml/trigger-training` - Start model training
- `GET /api/ml/training-status/{id}` - Check training progress

#### News & Sentiment
- `POST /api/news/sync` - Fetch news for stock
- `POST /api/news/sync-batch` - Batch news fetching

### **Flask ML Service (Port 8000)**

#### Inference
- `POST /api/predict` - Get portfolio recommendations

#### Training
- `POST /api/train/from-db` - Train with database data
- `GET /api/train/status/{id}` - Monitor training

#### Sentiment Analysis
- `POST /api/news/fetch` - Fetch news with sentiment
- `POST /api/news/analyze` - Analyze text sentiment

---

## 🔄 Data Flow

### **AI Trading Decision**
```
1. User triggers rebalancing
   ↓
2. Spring Boot fetches 60 days from PostgreSQL
   ↓
3. Calls Flask ML service with market data
   ↓
4. PPO model generates target weights
   ↓
5. Calculate required trades
   ↓
6. Execute orders with validation
   ↓
7. Update portfolio & log transactions
```

### **Automated Retraining (Quarterly)**
```
1. Scheduled task triggers (Jan/Apr/Jul/Oct)
   ↓
2. Export last 90 days from PostgreSQL
   ↓
3. Train 5 LSTM models (one per stock)
   ↓
4. Train PPO agent (500k timesteps)
   ↓
5. Save new models
   ↓
6. Next inference uses updated models
```

---

## 🧪 Testing

### Test Sentiment Analysis
```bash
curl -X POST http://localhost:8000/api/news/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple stock hits all-time high!"}'
```

### Test AI Rebalancing
```bash
curl -X POST http://localhost:8080/api/portfolio/1/rebalance
```

---

## 📈 Performance

- **Inference Speed:** <2 seconds
- **Training Time:** 2-3 hours (500k timesteps)
- **News Processing:** ~100 articles/second
- **Database Growth:** ~20 MB/year (with compression)

---

## 🛠️ Configuration

### Flask ML Service `.env`
```env
NEWS_API_KEY=your_newsapi_key_here
DEBUG=True
PORT=8000
```

### Spring Boot `application.properties`
```properties
spring.datasource.url=jdbc:postgresql://localhost:5432/trading_db
spring.datasource.username=postgres
spring.datasource.password=your_password
ai.service.url=http://localhost:8000
```

---

## 📝 License

MIT License - see LICENSE file for details

---

## 👥 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## 🙏 Acknowledgments

- **FinBERT** - ProsusAI/finbert
- **Stable-Baselines3** - DRL algorithms
- **Yahoo Finance** - Market data
- **NewsAPI** - Real-time news

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⭐ Star this repo if you find it useful!**
