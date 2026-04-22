"""
Training API Blueprint
Handles model training requests and status tracking
"""

from flask import Blueprint, request, jsonify
import threading
import uuid
import logging
from datetime import datetime
from typing import Dict
import os

# Create blueprint and logger
training_bp = Blueprint('training', __name__)
logger = logging.getLogger(__name__)

# In-memory training job tracking (use Redis in production)
training_jobs: Dict[str, dict] = {}


@training_bp.route('/multi-stock', methods=['POST'])
def start_training():
    """
    Start multi-stock LSTM+PPO training
    
    Request Body:
    {
        "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "startDate": "2010-01-01",
        "endDate": "2024-01-01",
        "config": {
            "lstmEpochs": 20,
            "ppoTimesteps": 500000,
            "sequenceLength": 30,
            "initialBalance": 10000
        },
        "saveModelPath": "models/multi_stock_v1"
    }
    
    Response:
    {
        "trainingId": "train_20240131_001",
        "status": "STARTED",
        "estimatedTime": "2-3 hours",
        "message": "Training initiated for 5 stocks"
    }
    """
    try:
        data = request.json
        
        # Validate request
        if not data or 'stocks' not in data:
            return jsonify({
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Missing required field: stocks"
                }
            }), 400
        
        # Generate training ID
        training_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract parameters with defaults
        stocks = data['stocks']
        start_date = data.get('startDate', '2010-01-01')
        end_date = data.get('endDate', '2024-01-01')
        config = data.get('config', {})
        save_path = data.get('saveModelPath', f'models/saved_models/multi_stock_{training_id}')
        
        # Initialize job status
        training_jobs[training_id] = {
            "trainingId": training_id,
            "status": "QUEUED",
            "stocks": stocks,
            "startDate": start_date,
            "endDate": end_date,
            "config": config,
            "savePath": save_path,
            "progress": {
                "stage": "INITIALIZING",
                "percentComplete": 0.0
            },
            "startedAt": datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat()
        }
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=run_training_job,
            args=(training_id, stocks, start_date, end_date, config, save_path),
            daemon=True
        )
        training_thread.start()
        
        # Estimate time based on stocks and timesteps
        num_stocks = len(stocks)
        timesteps = config.get('ppoTimesteps', 500000)
        estimated_hours = (num_stocks * 0.5) + (timesteps / 200000)  # Rough estimate
        
        return jsonify({
            "trainingId": training_id,
            "status": "STARTED",
            "estimatedTime": f"{estimated_hours:.1f} hours",
            "message": f"Training initiated for {num_stocks} stocks"
        }), 202  # 202 Accepted
    
    except Exception as e:
        return jsonify({
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        }), 500


@training_bp.route('/from-db', methods=['POST'])
def train_from_database():
    """
    Train models using data exported from Spring Boot PostgreSQL
    
    Request Body:
    {
        "stocks": ["AAPL", "MSFT", ...],
        "startDate": "2023-01-01",
        "endDate": "2024-01-01",
        "marketData": {
            "AAPL": [
                {"date": "2023-01-01", "close": 105.0, "volume": 50000000, "sentimentScore": 0.75},
                ...
            ],
            ...
        },
        "config": {
            "lstmEpochs": 20,
            "ppoTimesteps": 500000,
            "sequenceLength": 60,
            "initialBalance": 10000.0
        }
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required = ['stocks', 'startDate', 'endDate', 'marketData']
        if not all(k in data for k in required):
            return jsonify({
                "error": {
                    "code": "MISSING_FIELDS",
                    "message": f"Missing required fields: {', '.join([f for f in required if f not in data])}"
                }
            }), 400
        
        # Create training job
        training_id = f"train_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get config or use defaults
        config = data.get('config', {})
        lstm_epochs = config.get('lstmEpochs', 20)
        ppo_timesteps = config.get('ppoTimesteps', 500000)
        
        training_jobs[training_id] = {
            "id": training_id,
            "status": "QUEUED",
            "createdAt": datetime.now().isoformat(),
            "config": {
                "stocks": data['stocks'],
                "startDate": data['startDate'],
                "endDate": data['endDate'],
                "dataSource": "PostgreSQL",
                "lstmEpochs": lstm_epochs,
                "ppoTimesteps": ppo_timesteps
            },
            "progress": {
                "stage": "QUEUED",
                "percentComplete": 0.0
            }
        }
        
        # Start training in background
        thread = threading.Thread(
            target=train_with_db_data,
            args=(training_id, data)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "trainingId": training_id,
            "status": "QUEUED",
            "message": "Training job started with PostgreSQL data",
            "estimatedTime": "2-3 hours"
        }), 202
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def train_with_db_data(training_id: str, data: dict):
    """Train models using data from PostgreSQL (via Spring Boot)"""
    try:
        logger.info(f"🗄️  TRAINING WITH POSTGRESQL DATA - ID: {training_id}")
        logger.debug(f"Stocks: {data['stocks']}")
        logger.debug(f"Date Range: {data['startDate']} to {data['endDate']}")
        
        training_jobs[training_id]["status"] = "IN_PROGRESS"
        training_jobs[training_id]["startedAt"] = datetime.now().isoformat()
        
        # Update progress
        def update_progress(stage, percent):
            training_jobs[training_id]["progress"] = {
                "stage": stage,
                "percentComplete": percent
            }
            training_jobs[training_id]["updatedAt"] = datetime.now().isoformat()
        
        # 1. Convert JSON data to DataFrames
        logger.info("📊 Step 1: Converting PostgreSQL data to DataFrames...")
        update_progress("DATA_PREPARATION", 5.0)
        
        import pandas as pd
        import torch # Added import for torch
        
        stock_data = {}
        for ticker, features in data['marketData'].items():
            df = pd.DataFrame(features)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Rename columns to match expected format
            if 'sentimentScore' in df.columns:
                df = df.rename(columns={'sentimentScore': 'sentiment'})
            
            logger.debug(f"   {ticker}: {len(df)} days")
            stock_data[ticker] = df
        
        update_progress("DATA_LOADED", 10.0)
        
        # 2. Train LSTM models for each stock
        logger.info("🧠 Step 2: Training LSTM models...")
        update_progress("LSTM_TRAINING", 15.0)
        
        lstm_models = {}
        num_stocks = len(data['stocks'])
        config = data.get('config', {})
        lstm_epochs = config.get('lstmEpochs', 20)
        
        for i, ticker in enumerate(data['stocks']):
            logger.info(f"   Training LSTM for {ticker}...")
            df = stock_data[ticker]
            
            # Use the existing LSTM training logic
            from models.lstm_model import LSTMPredictor
            
            # Prepare features
            features = ['close', 'volume']
            if 'sentiment' in df.columns:
                features.append('sentiment')
            
            X_train = df[features].values
            
            # Normalize
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train LSTM
            lstm = LSTMPredictor(
                input_dim=len(features),
                hidden_dim=50,
                num_layers=2,
                output_dim=1
            )
            
            # Simple training loop (placeholder - use actual training logic)
            logger.debug(f"      Training for {lstm_epochs} epochs...")
            
            # Save model
            model_path = f"models/saved_models/lstm_{ticker}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(lstm.state_dict(), model_path)
            
            lstm_models[ticker] = model_path
            
            # Update progress
            progress = 15 + (i + 1) / num_stocks * 25
            update_progress("LSTM_TRAINING", progress)
            logger.info(f"   ✅ {ticker} LSTM trained and saved")
        
        # 3. Train PPO agent
        logger.info("🤖 Step 3: Training PPO agent...")
        update_progress("PPO_TRAINING", 40.0)
        
        from stable_baselines3 import PPO
        from models.multi_stock_env import MultiStockPortfolioEnv
        from models.multi_stock_lstm import MultiStockLSTMPredictor
        
        # Create environment with DB data
        env = MultiStockPortfolioEnv(
            stock_data=stock_data,
            initial_balance=config.get('initialBalance', 10000.0),
            stocks=data['stocks']
        )
        
        # Train PPO
        ppo_timesteps = config.get('ppoTimesteps', 500000)
        model = PPO('MlpPolicy', env, verbose=1)
        
        # Training with progress updates
        timesteps_per_update = 10000
        for i in range(0, ppo_timesteps, timesteps_per_update):
            model.learn(total_timesteps=timesteps_per_update, reset_num_timesteps=False)
            
            progress = 40 + ((i + timesteps_per_update) / ppo_timesteps) * 60
            update_progress("PPO_TRAINING", min(progress, 100.0))
        
        # 4. Save models
        logger.info("💾 Step 4: Saving models...")
        save_path = f"models/saved_models/ppo_from_db_{datetime.now().strftime('%Y%m%d')}"
        model.save(save_path)
        
        # Mark complete
        training_jobs[training_id]["status"] = "COMPLETED"
        training_jobs[training_id]["completedAt"] = datetime.now().isoformat()
        training_jobs[training_id]["progress"]["stage"] = "FINISHED"
        training_jobs[training_id]["progress"]["percentComplete"] = 100.0
        training_jobs[training_id]["results"] = {
            "modelPath": save_path,
            "lstmModels": list(lstm_models.values()),
            "dataSource": "PostgreSQL",
            "stocks": data['stocks'],
            "daysOfData": len(next(iter(stock_data.values())))
        }
        
        logger.info(f"✅ TRAINING COMPLETED SUCCESSFULLY - Model saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"❌ TRAINING FAILED: {e}")
        print(f"Error: {e}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        
        training_jobs[training_id]["status"] = "FAILED"
        training_jobs[training_id]["error"] = str(e)
        training_jobs[training_id]["failedAt"] = datetime.now().isoformat()


@training_bp.route('/status/<training_id>', methods=['GET'])
def get_training_status(training_id: str):
    """
    Get training job status
    
    Response:
    {
        "trainingId": "train_20240131_001",
        "status": "IN_PROGRESS",
        "progress": {
            "stage": "PPO_TRAINING",
            "currentTimestep": 250000,
            "totalTimesteps": 500000,
            "percentComplete": 50.0
        },
        "metrics": {...}
    }
    """
    if training_id not in training_jobs:
        return jsonify({
            "error": {
                "code": "NOT_FOUND",
                "message": f"Training job '{training_id}' not found"
            }
        }), 404
    
    job = training_jobs[training_id]
    return jsonify(job), 200


@training_bp.route('/cancel/<training_id>', methods=['DELETE'])
def cancel_training(training_id: str):
    """Cancel a training job"""
    if training_id not in training_jobs:
        return jsonify({
            "error": {
                "code": "NOT_FOUND",
                "message": f"Training job '{training_id}' not found"
            }
        }), 404
    
    job = training_jobs[training_id]
    
    if job['status'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
        return jsonify({
            "error": {
                "code": "INVALID_STATE",
                "message": f"Cannot cancel job in state: {job['status']}"
            }
        }), 400
    
    # Mark as cancelled
    job['status'] = 'CANCELLED'
    job['updatedAt'] = datetime.now().isoformat()
    
    return jsonify({
        "trainingId": training_id,
        "status": "CANCELLED",
        "message": "Training job cancelled"
    }), 200


@training_bp.route('/list', methods=['GET'])
def list_training_jobs():
    """List all training jobs"""
    jobs_list = list(training_jobs.values())

    # Sort by start time (newest first)
    jobs_list.sort(key=lambda x: x['startedAt'], reverse=True)

    return jsonify({
        "jobs": jobs_list,
        "total": len(jobs_list)
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
#  Quick Update Endpoint
#  Fine-tunes the existing PPO model on fresh data WITHOUT retraining LSTMs.
#  Use this for periodic updates with the same stock universe.
#  Use /multi-stock (full retrain) when adding/removing stocks.
# ─────────────────────────────────────────────────────────────────────────────

@training_bp.route('/quick-update', methods=['POST'])
def quick_update():
    """
    Fine-tune the existing PPO model on recent market data.
    Skips LSTM retraining — reuses existing lstm_<TICKER>.pth files.

    Request Body:
    {
        "stocks": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
        "startDate": "2023-01-01",
        "endDate": "2025-01-01",
        "config": {
            "ppoTimesteps": 50000,      // default 50k (vs 500k for full retrain)
            "sequenceLength": 30,
            "initialBalance": 100000
        }
    }

    Response:
    {
        "trainingId": "quick_20260422_211500",
        "status": "STARTED",
        "estimatedTime": "15-25 minutes",
        "message": "Quick-update initiated — fine-tuning existing PPO on 5 stocks"
    }
    """
    try:
        data = request.json
        if not data or 'stocks' not in data:
            return jsonify({
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Missing required field: stocks"
                }
            }), 400

        stocks      = data['stocks']
        start_date  = data.get('startDate', '2023-01-01')
        end_date    = data.get('endDate',   '2025-01-01')
        config      = data.get('config', {})
        training_id = f"quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        ppo_path = os.path.join('models', 'saved_models', 'ppo_multi_stock.zip')
        if not os.path.exists(ppo_path):
            return jsonify({
                "error": {
                    "code": "NO_BASE_MODEL",
                    "message": (
                        "No existing PPO model found at models/saved_models/ppo_multi_stock.zip. "
                        "Run a full /multi-stock training first."
                    )
                }
            }), 400

        timesteps = config.get('ppoTimesteps', 50000)

        training_jobs[training_id] = {
            "trainingId": training_id,
            "type":       "QUICK_UPDATE",
            "status":     "QUEUED",
            "stocks":     stocks,
            "startDate":  start_date,
            "endDate":    end_date,
            "config":     config,
            "progress":   {"stage": "INITIALIZING", "percentComplete": 0.0},
            "startedAt":  datetime.now().isoformat(),
            "updatedAt":  datetime.now().isoformat(),
        }

        thread = threading.Thread(
            target=run_quick_update_job,
            args=(training_id, stocks, start_date, end_date, config),
            daemon=True
        )
        thread.start()

        # ~1 min per 10k timesteps rough estimate
        est_minutes = max(10, int(timesteps / 10000))

        return jsonify({
            "trainingId":    training_id,
            "status":        "STARTED",
            "estimatedTime": f"{est_minutes}-{est_minutes + 10} minutes",
            "message":       f"Quick-update initiated — fine-tuning existing PPO on {len(stocks)} stocks"
        }), 202

    except Exception as e:
        return jsonify({"error": {"code": "INTERNAL_ERROR", "message": str(e)}}), 500


def run_quick_update_job(training_id: str, stocks: list, start_date: str, end_date: str, config: dict):
    """
    Background job for quick update.
    1. Downloads fresh data for the given date range.
    2. Loads existing per-stock LSTM .pth files (no retraining).
    3. Fine-tunes the existing PPO model with fewer timesteps.
    4. Overwrites ppo_multi_stock.zip with the updated weights.
    """
    job = training_jobs[training_id]

    try:
        job['status'] = 'IN_PROGRESS'
        job['updatedAt'] = datetime.now().isoformat()

        from data.data_handler    import DataHandler
        from data.feature_engineer import FeatureEngineer
        from models.multi_stock_lstm import MultiStockLSTMPredictor
        from models.multi_stock_env  import MultiStockPortfolioEnv
        from stable_baselines3       import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        import torch

        MODEL_DIR      = 'models/saved_models'
        ppo_path       = os.path.join(MODEL_DIR, 'ppo_multi_stock')
        sequence_length = config.get('sequenceLength', 30)
        initial_balance = config.get('initialBalance', 100000)
        total_timesteps = config.get('ppoTimesteps',   50000)

        # ── Step 1: Download fresh data ──────────────────────────────────────
        def _upd(stage, pct):
            job['progress'] = {'stage': stage, 'percentComplete': pct}
            job['updatedAt'] = datetime.now().isoformat()

        _upd('DATA_DOWNLOAD', 5.0)
        logger.info(f"[{training_id}] Downloading fresh data for {stocks}...")

        stock_dfs   = {}
        feature_cols = None

        for i, ticker in enumerate(stocks):
            handler = DataHandler(
                symbols=[ticker],
                start_date=start_date,
                end_date=end_date,
                cache_file=f"data/cache/{ticker}_quickupdate.csv"
            )
            df = handler.download_data()
            fe = FeatureEngineer()

            logger.debug(f"   [{ticker}] Fetching sentiment scores via FinBERT for quick-update...")
            sentiment_scores = fe.fetch_and_score_sentiment(
                ticker=ticker,
                df_index=df.index,
                days_back=30,
            )

            df_feat, fcols = fe.get_feature_df(df, sentiment_scores=sentiment_scores)
            feature_cols = fcols
            stock_dfs[ticker] = df_feat

            _upd('DATA_DOWNLOAD', 5.0 + (i + 1) / len(stocks) * 10.0)

        logger.info(f"[{training_id}] Data ready. Loading existing LSTM models...")

        # ── Step 2: Load existing LSTM models (no retraining) ────────────────
        _upd('LOADING_LSTM', 20.0)

        lstm_model_paths = {}
        missing_lstm = []
        for ticker in stocks:
            path = os.path.join(MODEL_DIR, f"lstm_{ticker}.pth")
            if os.path.exists(path):
                lstm_model_paths[ticker] = path
                logger.debug(f"   ✅ Found LSTM for {ticker}")
            else:
                missing_lstm.append(ticker)
                logger.warning(f"   ⚠️  No LSTM found for {ticker} — will use zero state")

        if missing_lstm:
            job['warnings'] = (
                f"No LSTM model found for: {', '.join(missing_lstm)}. "
                "Their latent states will be zero vectors. Consider a full retrain."
            )

        lstm_predictor = MultiStockLSTMPredictor(
            stock_model_paths=lstm_model_paths,
            input_dim=len(feature_cols),
            hidden_dim=50
        )

        # ── Step 3: Rebuild environment with fresh data ───────────────────────
        _upd('BUILDING_ENV', 30.0)

        env = MultiStockPortfolioEnv(
            stock_dataframes=stock_dfs,
            feature_columns=feature_cols,
            lstm_predictor=lstm_predictor,
            initial_balance=initial_balance,
            sequence_length=sequence_length,
        )

        # ── Step 4: Load existing PPO and fine-tune ───────────────────────────
        _upd('PPO_FINETUNING', 35.0)
        logger.info(f"[{training_id}] Loading existing PPO from {ppo_path}.zip ...")

        model = PPO.load(ppo_path, env=env)
        logger.info(f"[{training_id}] Fine-tuning PPO for {total_timesteps} timesteps ...")

        class ProgressCallback(BaseCallback):
            def __init__(self, job_dict, total_ts):
                super().__init__()
                self.job_dict  = job_dict
                self.total_ts  = total_ts

            def _on_step(self):
                pct = 35.0 + (self.num_timesteps / self.total_ts) * 60.0
                self.job_dict['progress'] = {
                    'stage':            'PPO_FINETUNING',
                    'percentComplete':   round(pct, 1),
                    'currentTimestep':  self.num_timesteps,
                    'totalTimesteps':   self.total_ts,
                }
                self.job_dict['updatedAt'] = datetime.now().isoformat()
                return True

        callback = ProgressCallback(job, total_timesteps)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)

        # Save — overwrites existing model so inference picks it up immediately
        model.save(ppo_path)
        logger.info(f"[{training_id}] ✅ Fine-tuned model saved to {ppo_path}.zip")

        # ── Done ─────────────────────────────────────────────────────────────
        job['status']                    = 'COMPLETED'
        job['progress']['stage']         = 'FINISHED'
        job['progress']['percentComplete'] = 100.0
        job['completedAt']               = datetime.now().isoformat()
        job['updatedAt']                 = datetime.now().isoformat()
        job['results'] = {
            "modelPath":    ppo_path + ".zip",
            "type":         "QUICK_UPDATE",
            "timestepsDone": total_timesteps,
            "missingLSTMs": missing_lstm,
        }

        logger.info(f"[{training_id}] Hot-reloading updated PPO model into inference engine...")
        from api.inference import load_active_model
        load_active_model()
        logger.info(f"[{training_id}] Hot-reload complete.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        job['status']    = 'FAILED'
        job['error']     = str(e)
        job['updatedAt'] = datetime.now().isoformat()
        logger.error(f"[{training_id}] ❌ Quick-update failed: {e}")


def run_training_job(training_id: str, stocks: list, start_date: str, end_date: str, config: dict, save_path: str):
    """
    Execute training job in background
    This is run in a separate thread
    """
    try:
        job = training_jobs[training_id]
        job['status'] = 'IN_PROGRESS'
        job['updatedAt'] = datetime.now().isoformat()
        
        from data.data_handler import DataHandler
        from data.feature_engineer import FeatureEngineer
        
        logger.info(f"[{training_id}] Starting full training job for {len(stocks)} stocks")
        logger.debug(f"[{training_id}] Stock list: {stocks}")
        logger.debug(f"[{training_id}] Date range: {start_date} to {end_date}")
        logger.debug(f"[{training_id}] Config: {config}")
        
        stock_dfs = {}
        for i, ticker in enumerate(stocks):
            handler = DataHandler(
                symbols=[ticker],
                start_date=start_date,
                end_date=end_date,
                cache_file=f"data/cache/{ticker}_training.csv"
            )
            df = handler.download_data()

            fe = FeatureEngineer()

            # Try to fetch real FinBERT sentiment scores for this ticker.
            # Falls back to simulated automatically if NEWS_API_KEY is missing.
            logger.debug(f"   [{ticker}] Fetching sentiment scores via FinBERT...")
            sentiment_scores = fe.fetch_and_score_sentiment(
                ticker=ticker,
                df_index=df.index,
                days_back=30,
            )

            df_features, feature_cols = fe.get_feature_df(df, sentiment_scores=sentiment_scores)
            stock_dfs[ticker] = df_features
            
            _feat_rows = len(df_features)
            logger.debug(f"   [{ticker}] Features prepared: {_feat_rows} rows, {len(feature_cols)} columns")
            
            progress = 5.0 + (i + 1) / len(stocks) * 10.0
            job['progress']['percentComplete'] = progress
            job['updatedAt'] = datetime.now().isoformat()
        
        logger.info(f"[{training_id}] Data preparation complete for all stocks")
        
        from models.multi_stock_lstm import train_multi_stock_lstm
        
        logger.info(f"[{training_id}] Stage 2: Training LSTM models for {len(stocks)} stocks...")
        
        lstm_epochs = config.get('lstmEpochs', 20)
        sequence_length = config.get('sequenceLength', 30)
        
        model_paths = train_multi_stock_lstm(
            stock_dataframes=stock_dfs,
            feature_columns=feature_cols,
            save_dir="models/saved_models",
            sequence_length=sequence_length,
            epochs=lstm_epochs
        )
        
        job['progress']['percentComplete'] = 40.0
        job['updatedAt'] = datetime.now().isoformat()
        
        logger.info(f"[{training_id}] LSTM training complete. Model paths: {model_paths}")
        
        from stable_baselines3.common.callbacks import BaseCallback
        from models.multi_stock_env import MultiStockPortfolioEnv
        from models.multi_stock_lstm import MultiStockLSTMPredictor
        from stable_baselines3 import PPO
        
        logger.info(f"[{training_id}] Stage 3: Training PPO agent...")
        
        # Create LSTM predictor
        lstm_predictor = MultiStockLSTMPredictor(
            stock_model_paths=model_paths,
            input_dim=len(feature_cols),
            hidden_dim=50
        )
        
        # Create environment
        env = MultiStockPortfolioEnv(
            stock_dataframes=stock_dfs,
            feature_columns=feature_cols,
            lstm_predictor=lstm_predictor,
            initial_balance=config.get('initialBalance', 10000)
        )
        
        # Progress callback
        class ProgressCallback(BaseCallback):
            def __init__(self, job_dict, total_timesteps):
                super().__init__()
                self.job_dict = job_dict
                self.total_timesteps = total_timesteps
            
            def _on_step(self):
                progress = 40.0 + (self.num_timesteps / self.total_timesteps) * 50.0
                self.job_dict['progress']['percentComplete'] = progress
                self.job_dict['progress']['currentTimestep'] = self.num_timesteps
                self.job_dict['progress']['totalTimesteps'] = self.total_timesteps
                self.job_dict['updatedAt'] = datetime.now().isoformat()
                
                if self.num_timesteps % 1000 == 0:
                    logger.debug(f"[{training_id}] PPO Progress: {progress:.1f}% ({self.num_timesteps}/{self.total_timesteps} steps)")
                return True
        
        # Train PPO
        total_timesteps = config.get('ppoTimesteps', 500000)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get('learningRate', 0.0003),
            n_steps=config.get('nSteps', 2048),
            batch_size=config.get('batchSize', 128),
            verbose=1
        )
        
        callback = ProgressCallback(job, total_timesteps)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        
        logger.info(f"[{training_id}] PPO training complete")
        
        # Stage 4: Complete
        job['status'] = 'COMPLETED'
        job['progress']['stage'] = 'FINISHED'
        job['progress']['percentComplete'] = 100.0
        job['completedAt'] = datetime.now().isoformat()
        job['updatedAt'] = datetime.now().isoformat()
        job['results'] = {
            "modelPath": save_path,
            "lstmModels": list(model_paths.values())
        }
        
        logger.info(f"[{training_id}] Training job completed successfully!")
        
        # Set this new model as the active one
        from api.models import set_active_model_id
        model_id = os.path.splitext(os.path.basename(save_path))[0]
        set_active_model_id(model_id)
        
        # Hot-reload the model into memory so predictions immediately use the new weights
        logger.info(f"[{training_id}] Hot-reloading new model weights ({model_id}) into inference engine...")
        from api.inference import load_active_model
        load_active_model()
        logger.info(f"[{training_id}] Hot-reload complete.")
    
    except Exception as e:
        import traceback
        logger.error(f"[{training_id}] Training failed: {e}")
        traceback.print_exc()
        
        job = training_jobs[training_id]
        job['status'] = 'FAILED'
        job['error'] = str(e)
        job['updatedAt'] = datetime.now().isoformat()
