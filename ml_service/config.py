"""
Configuration settings for ML Service
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""
    
    # Flask Settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', 8000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Model Settings
    MODEL_DIR = os.getenv('MODEL_DIR', 'models/saved_models')
    CACHE_DIR = os.getenv('CACHE_DIR', 'data/cache')
    
    # LSTM Configuration
    LSTM_SEQUENCE_LENGTH = int(os.getenv('LSTM_SEQUENCE_LENGTH', 30))
    LSTM_HIDDEN_DIM = int(os.getenv('LSTM_HIDDEN_DIM', 50))
    LSTM_NUM_LAYERS = int(os.getenv('LSTM_NUM_LAYERS', 2))
    LSTM_EPOCHS = int(os.getenv('LSTM_EPOCHS', 20))
    LSTM_BATCH_SIZE = int(os.getenv('LSTM_BATCH_SIZE', 32))
    
    # PPO Configuration
    PPO_LEARNING_RATE = float(os.getenv('PPO_LEARNING_RATE', 0.0003))
    PPO_N_STEPS = int(os.getenv('PPO_N_STEPS', 2048))
    PPO_BATCH_SIZE = int(os.getenv('PPO_BATCH_SIZE', 128))
    PPO_TOTAL_TIMESTEPS = int(os.getenv('PPO_TOTAL_TIMESTEPS', 500000))
    
    # Data Settings
    DATA_START_DATE = os.getenv('DATA_START_DATE', '2010-01-01')
    DATA_END_DATE = os.getenv('DATA_END_DATE', '2024-01-01')
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 10000))
    
    # API Settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create directories if they don't exist
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
