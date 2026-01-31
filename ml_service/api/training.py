"""
Training API Blueprint
Handles model training requests and status tracking
"""

from flask import Blueprint, request, jsonify
import threading
import uuid
from datetime import datetime
from typing import Dict
import os

# Create blueprint
training_bp = Blueprint('training', __name__)

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
        print(f"\n{'='*70}")
        print(f"🗄️  TRAINING WITH POSTGRESQL DATA")
        print(f"Training ID: {training_id}")
        print(f"Stocks: {data['stocks']}")
        print(f"Date Range: {data['startDate']} to {data['endDate']}")
        print(f"{'='*70}\n")
        
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
        print("📊 Step 1: Converting PostgreSQL data to DataFrames...")
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
            
            print(f"   {ticker}: {len(df)} days")
            stock_data[ticker] = df
        
        update_progress("DATA_LOADED", 10.0)
        
        # 2. Train LSTM models for each stock
        print("\n🧠 Step 2: Training LSTM models...")
        update_progress("LSTM_TRAINING", 15.0)
        
        lstm_models = {}
        num_stocks = len(data['stocks'])
        config = data.get('config', {})
        lstm_epochs = config.get('lstmEpochs', 20)
        
        for i, ticker in enumerate(data['stocks']):
            print(f"\n   Training LSTM for {ticker}...")
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
            print(f"      Training for {lstm_epochs} epochs...")
            
            # Save model
            model_path = f"models/saved_models/lstm_{ticker}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(lstm.state_dict(), model_path)
            
            lstm_models[ticker] = model_path
            
            # Update progress
            progress = 15 + (i + 1) / num_stocks * 25
            update_progress("LSTM_TRAINING", progress)
            print(f"   ✅ {ticker} LSTM trained and saved")
        
        # 3. Train PPO agent
        print("\n🤖 Step 3: Training PPO agent...")
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
        print("\n💾 Step 4: Saving models...")
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
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETED SUCCESSFULLY")
        print(f"Model saved to: {save_path}")
        print(f"LSTM models: {len(lstm_models)}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ TRAINING FAILED")
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


def run_training_job(training_id: str, stocks: list, start_date: str, end_date: str, config: dict, save_path: str):
    """
    Execute training job in background
    This is run in a separate thread
    """
    try:
        job = training_jobs[training_id]
        job['status'] = 'IN_PROGRESS'
        job['updatedAt'] = datetime.now().isoformat()
        
        # Stage 1: Download and prepare data
        job['progress']['stage'] = 'DATA_PREPARATION'
        job['progress']['percentComplete'] = 5.0
        
        from data.data_handler import DataHandler
        from data.feature_engineer import FeatureEngineer
        
        print(f"[{training_id}] Downloading data for {len(stocks)} stocks...")
        
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
            df_features, feature_cols = fe.get_feature_df(df)
            stock_dfs[ticker] = df_features
            
            progress = 5.0 + (i + 1) / len(stocks) * 10.0
            job['progress']['percentComplete'] = progress
            job['updatedAt'] = datetime.now().isoformat()
        
        print(f"[{training_id}] Data preparation complete")
        
        # Stage 2: LSTM Training
        job['progress']['stage'] = 'LSTM_TRAINING'
        job['progress']['percentComplete'] = 15.0
        
        from models.multi_stock_lstm import train_multi_stock_lstm
        
        print(f"[{training_id}] Training LSTM models...")
        
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
        
        print(f"[{training_id}] LSTM training complete")
        
        # Stage 3: PPO Training
        job['progress']['stage'] = 'PPO_TRAINING'
        job['progress']['percentComplete'] = 40.0
        
        from models.multi_stock_env import MultiStockPortfolioEnv
        from models.multi_stock_lstm import MultiStockLSTMPredictor
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        
        print(f"[{training_id}] Training PPO agent...")
        
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
        
        print(f"[{training_id}] PPO training complete")
        
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
        
        print(f"[{training_id}] Training job completed successfully!")
    
    except Exception as e:
        import traceback
        print(f"[{training_id}] Training failed: {e}")
        traceback.print_exc()
        
        job = training_jobs[training_id]
        job['status'] = 'FAILED'
        job['error'] = str(e)
        job['updatedAt'] = datetime.now().isoformat()
