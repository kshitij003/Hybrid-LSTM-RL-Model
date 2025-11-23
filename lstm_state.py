"""
Module 3: LSTM Predictive State Module
Defines, trains (Stage 1), and hosts the LSTM model for inference.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import os
from typing import List


class LSTMStateModel(nn.Module):
    """
    Defines the LSTM architecture.
    """
    def __init__(self, input_dim: int, hidden_dim: int, 
                 num_layers: int, output_dim: int = 1):
        super(LSTMStateModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        prediction = self.fc(lstm_out[:, -1, :])

        return prediction, h_n


def create_sequences(df: pd.DataFrame, feature_cols: List[str],
                     sequence_length: int, target_col: str = "Close"):
    """
    Creates sequences for LSTM training.
    """
    X = []
    y = []

    feature_data = df[feature_cols].values
    target_data = df[target_col].values

    for i in range(len(df) - sequence_length):
        X.append(feature_data[i:i + sequence_length])
        y.append(target_data[i + sequence_length])

    return np.array(X), np.array(y).reshape(-1, 1)


def train_lstm_model(
    train_df: pd.DataFrame, 
    feature_cols: List[str],
    target_col: str = 'Close',
    sequence_length: int = 30,
    hidden_dim: int = 50,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 32,
    model_save_path: str = 'models/lstm_state.pth'):

    print("\n--- Starting Stage 1: LSTM Pre-training ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create sequences
    X_train, y_train = create_sequences(train_df, feature_cols, sequence_length, target_col)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(feature_cols)
    model = LSTMStateModel(input_dim, hidden_dim, num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            prediction, _ = model(X_batch)
            loss = criterion(prediction, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / len(train_loader))

    print("LSTM Pre-training finished.")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained LSTM model saved to: {model_save_path}")


class LSTMPredictor:
    """
    Load the pre-trained LSTM and output latent states for RL.
    """
    def __init__(self, model_path: str, input_dim: int, 
                 hidden_dim: int, num_layers: int):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LSTMStateModel(input_dim, hidden_dim, num_layers).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def get_latent_state(self, sequence_data: np.ndarray) -> np.ndarray:

        sequence_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, h_n = self.model(sequence_tensor)

        latent_state = h_n[-1].cpu().numpy().flatten()

        return latent_state
