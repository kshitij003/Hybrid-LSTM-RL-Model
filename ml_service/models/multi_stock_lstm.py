"""
Multi-Stock LSTM Predictor
Manages multiple LSTM models (one per stock) and provides latent states
"""

import numpy as np
import torch
import os
from typing import Dict, List
from models.lstm_model import LSTMStateModel, LSTMPredictor


class MultiStockLSTMPredictor:
    """
    Wrapper class to manage multiple LSTM models for multi-stock trading.
    Each stock has its own trained LSTM model.
    """
    
    def __init__(
        self,
        stock_model_paths: Dict[str, str],
        input_dim: int,
        hidden_dim: int = 50,
        num_layers: int = 2
    ):
        """
        Initialize multi-stock LSTM predictor.
        
        Args:
            stock_model_paths: Dict mapping {ticker: model_path}
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
        """
        self.stock_tickers = sorted(list(stock_model_paths.keys()))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Load all LSTM models
        self.predictors = {}
        for ticker, model_path in stock_model_paths.items():
            if os.path.exists(model_path):
                self.predictors[ticker] = LSTMPredictor(
                    model_path=model_path,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers
                )
                print(f" Loaded LSTM model for {ticker}")
            else:
                print(f"  Model not found for {ticker}: {model_path}")
                # Create dummy predictor for testing
                self.predictors[ticker] = None
    
    def get_latent_states(self, market_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get LSTM latent states for all stocks.
        
        Args:
            market_data: Dict of {ticker: sequence_array}
                        Each sequence_array is shape (sequence_length, num_features)
        
        Returns:
            Dict of {ticker: latent_state_array}
            Each latent_state_array is shape (hidden_dim,)
        """
        latent_states = {}
        
        for ticker in self.stock_tickers:
            if ticker not in market_data:
                # Return zeros if data missing
                latent_states[ticker] = np.zeros(self.hidden_dim, dtype=np.float32)
                continue
            
            predictor = self.predictors.get(ticker)
            
            if predictor is None:
                # No trained model - return zeros
                latent_states[ticker] = np.zeros(self.hidden_dim, dtype=np.float32)
            else:
                # Get latent state from LSTM
                sequence_data = market_data[ticker]
                latent_state = predictor.get_latent_state(sequence_data)
                latent_states[ticker] = latent_state
        
        return latent_states
    
    def predict_prices(self, market_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Predict next price for all stocks using LSTM.
        
        Args:
            market_data: Dict of {ticker: sequence_array}
        
        Returns:
            Dict of {ticker: predicted_price}
        """
        predictions = {}
        
        for ticker in self.stock_tickers:
            if ticker not in market_data:
                predictions[ticker] = 0.0
                continue
            
            predictor = self.predictors.get(ticker)
            
            if predictor is None:
                predictions[ticker] = 0.0
            else:
                sequence_data = market_data[ticker]
                sequence_tensor = torch.tensor(
                    sequence_data, dtype=torch.float32
                ).unsqueeze(0).to(predictor.device)
                
                with torch.no_grad():
                    prediction, _ = predictor.model(sequence_tensor)
                
                predictions[ticker] = float(prediction.cpu().numpy()[0, 0])
        
        return predictions
    
    @classmethod
    def create_dummy_predictor(cls, stock_tickers: List[str], input_dim: int, hidden_dim: int = 50):
        """
        Create a dummy predictor for testing without trained models.
        
        Args:
            stock_tickers: List of stock tickers
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
        
        Returns:
            MultiStockLSTMPredictor instance
        """
        # Create dummy model paths (won't be loaded)
        dummy_paths = {ticker: f"dummy_{ticker}.pth" for ticker in stock_tickers}
        
        predictor = cls(
            stock_model_paths=dummy_paths,
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        return predictor


class DummyLSTMPredictor:
    """
    Dummy LSTM predictor for testing without trained models.
    Returns random latent states.
    """
    
    def __init__(self, hidden_dim: int = 50):
        self.hidden_dim = hidden_dim
        self.device = torch.device('cpu')
    
    def get_latent_state(self, sequence_data: np.ndarray) -> np.ndarray:
        """Return random latent state for testing"""
        # Use last price as seed for reproducibility
        np.random.seed(int(sequence_data[-1, 0] * 1000) % 1000)
        latent_state = np.random.randn(self.hidden_dim).astype(np.float32)
        return latent_state


def train_multi_stock_lstm(
    stock_dataframes: Dict[str, any],
    feature_columns: List[str],
    save_dir: str = "models/saved_models",
    sequence_length: int = 30,
    hidden_dim: int = 50,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 32
):
    """
    Train LSTM models for multiple stocks.
    
    Args:
        stock_dataframes: Dict of {ticker: DataFrame}
        feature_columns: List of feature column names
        save_dir: Directory to save models
        Other args: LSTM training parameters
    
    Returns:
        Dict of {ticker: model_path}
    """
    from models.lstm_model import train_lstm_model
    
    os.makedirs(save_dir, exist_ok=True)
    model_paths = {}
    
    for ticker, df in stock_dataframes.items():
        print(f"\n{'='*60}")
        print(f"Training LSTM for {ticker}")
        print(f"{'='*60}")
        
        model_path = os.path.join(save_dir, f"lstm_{ticker}.pth")
        
        try:
            train_lstm_model(
                train_df=df,
                feature_cols=feature_columns,
                target_col='Close',
                sequence_length=sequence_length,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=epochs,
                batch_size=batch_size,
                model_save_path=model_path
            )
            
            model_paths[ticker] = model_path
            print(f" Successfully trained LSTM for {ticker}")
            
        except Exception as e:
            print(f" Failed to train LSTM for {ticker}: {e}")
    
    return model_paths
