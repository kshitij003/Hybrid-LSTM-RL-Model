"""
Multi-Stock Portfolio Trading Environment
Extends single-stock environment to handle portfolio of multiple stocks
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import deque


class MultiStockPortfolioEnv(gym.Env):
    """
    Gymnasium environment for multi-stock portfolio trading.
    
    The agent manages a portfolio of multiple stocks with the goal of
    maximizing returns while managing risk (volatility, drawdown).
    
    Action Space:
        Continuous Box: Portfolio weights for each stock + cash
        Shape: (num_stocks + 1,)
        Range: [0.0, 1.0] for each weight
        Constraint: Sum of all weights must equal 1.0
    
    Observation Space:
        Box containing:
        - LSTM latent states for each stock (if using LSTM)
        - Current prices for each stock (normalized)
        - Current portfolio weights
        - Portfolio value
        - Cash balance
        - Recent returns
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        stock_dataframes: Dict[str, pd.DataFrame],
        feature_columns: List[str],
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        lstm_predictor=None,  # Optional: MultiStockLSTMPredictor
        sequence_length: int = 30,
        reward_scaling: float = 1.0
    ):
        """
        Initialize multi-stock portfolio environment.
        
        Args:
            stock_dataframes: Dict of {ticker: DataFrame} with OHLCV + features
            feature_columns: List of feature column names
            initial_balance: Starting cash balance
            transaction_cost: Trading cost as fraction of trade value
            lstm_predictor: Optional LSTM predictor for enhanced observations
            sequence_length: Lookback period for LSTM
            reward_scaling: Scale factor for rewards
        """
        super().__init__()
        
        self.stock_tickers = sorted(list(stock_dataframes.keys()))
        self.num_stocks = len(self.stock_tickers)
        self.dfs = {ticker: df.reset_index(drop=True) for ticker, df in stock_dataframes.items()}
        self.feature_columns = feature_columns
        self.initial_balance = float(initial_balance)
        self.transaction_cost = transaction_cost
        self.lstm_predictor = lstm_predictor
        self.sequence_length = sequence_length
        self.reward_scaling = reward_scaling
        
        # Verify all dataframes have same length
        lengths = [len(df) for df in self.dfs.values()]
        assert len(set(lengths)) == 1, "All stock dataframes must have same length"
        self.max_steps = lengths[0] - sequence_length - 1
        
        # Action space: Portfolio weights (must sum to 1.0)
        # [weight_stock1, weight_stock2, ..., weight_stockN, weight_cash]
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_stocks + 1,),
            dtype=np.float32
        )
        
        # Observation space calculation
        obs_dim = self._calculate_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State variables (initialized in reset)
        self.step_idx = 0
        self.cash = 0.0
        self.shares = {}  # {ticker: num_shares}
        self.portfolio_value = 0.0
        self.portfolio_weights = np.zeros(self.num_stocks + 1)
        self.equity_history = deque(maxlen=100)
        self.returns_history = deque(maxlen=20)
        
    def _calculate_obs_dim(self) -> int:
        """Calculate observation space dimensionality"""
        dim = 0
        
        # LSTM latent states (if available)
        if self.lstm_predictor is not None:
            lstm_dim = 50  # Assuming 50-dim hidden state per stock
            dim += lstm_dim * self.num_stocks
        
        # Current prices (normalized)
        dim += self.num_stocks
        
        # Current portfolio weights (num_stocks + cash)
        dim += self.num_stocks + 1
        
        # Portfolio state (value, cash, total equity)
        dim += 3
        
        # Recent returns (last 5 periods)
        dim += 5
        
        # Sharpe ratio, max drawdown
        dim += 2
        
        return dim
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.step_idx = self.sequence_length  # Start after sequence
        self.cash = self.initial_balance
        self.shares = {ticker: 0 for ticker in self.stock_tickers}
        self.portfolio_value = self.initial_balance
        
        # Initialize weights: 100% cash
        self.portfolio_weights = np.zeros(self.num_stocks + 1, dtype=np.float32)
        self.portfolio_weights[-1] = 1.0  # All cash initially
        
        self.equity_history.clear()
        self.equity_history.append(self.initial_balance)
        
        self.returns_history.clear()
        self.returns_history.extend([0.0] * 5)
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        obs_components = []
        
        # 1. LSTM latent states (if available)
        if self.lstm_predictor is not None:
            lstm_states = self._get_lstm_states()
            obs_components.append(lstm_states)
        
        # 2. Current prices (normalized by initial price)
        prices = self._get_current_prices()
        first_prices = self._get_prices_at_step(self.sequence_length)
        normalized_prices = prices / (first_prices + 1e-8)
        obs_components.append(normalized_prices)
        
        # 3. Current portfolio weights
        obs_components.append(self.portfolio_weights)
        
        # 4. Portfolio state
        total_value = self._calculate_total_value()
        portfolio_state = np.array([
            total_value / self.initial_balance,  # Normalized total value
            self.cash / self.initial_balance,     # Normalized cash
            (total_value - self.initial_balance) / self.initial_balance  # Return
        ], dtype=np.float32)
        obs_components.append(portfolio_state)
        
        # 5. Recent returns
        recent_returns = np.array(list(self.returns_history)[-5:], dtype=np.float32)
        obs_components.append(recent_returns)
        
        # 6. Sharpe ratio and max drawdown
        sharpe = self._calculate_sharpe_ratio()
        max_dd = self._calculate_max_drawdown()
        risk_metrics = np.array([sharpe, max_dd], dtype=np.float32)
        obs_components.append(risk_metrics)
        
        # Concatenate all components
        observation = np.concatenate(obs_components)
        
        # Clip and handle NaN
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        observation = np.clip(observation, -10.0, 10.0)
        
        return observation
    
    def _get_lstm_states(self) -> np.ndarray:
        """Get LSTM latent states for all stocks"""
        if self.lstm_predictor is None:
            return np.array([], dtype=np.float32)
        
        # Get sequence data for each stock
        market_data = {}
        for ticker in self.stock_tickers:
            start_idx = max(0, self.step_idx - self.sequence_length)
            end_idx = self.step_idx
            sequence_data = self.dfs[ticker].loc[start_idx:end_idx-1, self.feature_columns].values
            market_data[ticker] = sequence_data
        
        # Get latent states from LSTM predictor
        lstm_states = self.lstm_predictor.get_latent_states(market_data)
        
        # Concatenate all latent states
        states_list = [lstm_states[ticker] for ticker in self.stock_tickers]
        return np.concatenate(states_list)
    
    def _get_current_prices(self) -> np.ndarray:
        """Get current close prices for all stocks"""
        prices = np.array([
            float(self.dfs[ticker]["Close"].iloc[self.step_idx])
            for ticker in self.stock_tickers
        ], dtype=np.float32)
        return np.maximum(prices, 0.01)  # Prevent zero prices
    
    def _get_prices_at_step(self, step: int) -> np.ndarray:
        """Get prices at specific step"""
        prices = np.array([
            float(self.dfs[ticker]["Close"].iloc[step])
            for ticker in self.stock_tickers
        ], dtype=np.float32)
        return np.maximum(prices, 0.01)
    
    def _calculate_total_value(self) -> float:
        """Calculate total portfolio value"""
        prices = self._get_current_prices()
        stock_values = sum(
            self.shares[ticker] * prices[i]
            for i, ticker in enumerate(self.stock_tickers)
        )
        return self.cash + stock_values
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Portfolio weights [stock1, stock2, ..., stockN, cash]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to ensure sum = 1.0
        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            # If all zeros, keep current allocation
            action = self.portfolio_weights.copy()
        
        # Store previous value
        prev_value = self._calculate_total_value()
        
        # Execute rebalancing
        transaction_costs = self._rebalance_portfolio(action)
        
        # Move to next time step
        self.step_idx += 1
        
        # Calculate new value after price changes
        current_value = self._calculate_total_value()
        
        # Update portfolio weights based on actual positions
        self._update_portfolio_weights()
        
        # Calculate reward
        reward = self._calculate_reward(prev_value, current_value, transaction_costs)
        
        # Update history
        self.equity_history.append(current_value)
        period_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
        self.returns_history.append(period_return)
        
        # Check termination
        terminated = self.step_idx >= self.max_steps
        truncated = current_value < self.initial_balance * 0.1  # Stop if lost 90%
        
        # Info dict
        info = {
            "portfolio_value": current_value,
            "cash": self.cash,
            "return": (current_value - self.initial_balance) / self.initial_balance,
            "transaction_costs": transaction_costs,
            "weights": self.portfolio_weights.copy()
        }
        
        return self._get_observation(), float(reward), terminated, truncated, info
    
    def _rebalance_portfolio(self, target_weights: np.ndarray) -> float:
        """
        Rebalance portfolio to match target weights.
        
        Args:
            target_weights: Desired portfolio weights
        
        Returns:
            Total transaction costs incurred
        """
        current_value = self._calculate_total_value()
        prices = self._get_current_prices()
        
        total_cost = 0.0
        
        # Calculate target positions
        for i, ticker in enumerate(self.stock_tickers):
            target_value = target_weights[i] * current_value
            current_shares = self.shares[ticker]
            current_stock_value = current_shares * prices[i]
            
            # Calculate shares to trade
            value_diff = target_value - current_stock_value
            shares_to_trade = value_diff / prices[i]
            
            if abs(shares_to_trade) > 0.01:  # Minimum trade threshold
                # Execute trade
                trade_value = abs(shares_to_trade * prices[i])
                cost = trade_value * self.transaction_cost
                total_cost += cost
                
                if shares_to_trade > 0:  # Buy
                    cost_including_commission = shares_to_trade * prices[i] + cost
                    if self.cash >= cost_including_commission:
                        self.shares[ticker] += shares_to_trade
                        self.cash -= cost_including_commission
                else:  # Sell
                    shares_to_sell = min(abs(shares_to_trade), current_shares)
                    proceeds = shares_to_sell * prices[i] - cost
                    self.shares[ticker] -= shares_to_sell
                    self.cash += proceeds
        
        return total_cost
    
    def _update_portfolio_weights(self):
        """Update portfolio weights based on current positions"""
        total_value = self._calculate_total_value()
        if total_value <= 0:
            return
        
        prices = self._get_current_prices()
        
        for i, ticker in enumerate(self.stock_tickers):
            stock_value = self.shares[ticker] * prices[i]
            self.portfolio_weights[i] = stock_value / total_value
        
        # Cash weight
        self.portfolio_weights[-1] = self.cash / total_value
    
    def _calculate_reward(self, prev_value: float, current_value: float, transaction_costs: float) -> float:
        """
        Calculate reward with multiple components.
        
        Components:
        1. Portfolio return
        2. Sharpe ratio encouragement
        3. Drawdown penalty
        4. Transaction cost penalty
        """
        # 1. Return
        value_change = current_value - prev_value
        return_pct = value_change / prev_value if prev_value > 0 else 0.0
        
        # 2. Sharpe-like term
        if len(self.returns_history) >= 5:
            returns_array = np.array(list(self.returns_history))
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array) + 1e-6
            sharpe_reward = mean_return / std_return
        else:
            sharpe_reward = 0.0
        
        # 3. Drawdown penalty
        max_drawdown = self._calculate_max_drawdown()
        drawdown_penalty = -max_drawdown * 0.5
        
        # 4. Transaction cost penalty
        cost_penalty = -transaction_costs / prev_value if prev_value > 0 else 0.0
        
        # Combined reward
        reward = (
            return_pct * 10.0 +           # Main signal: returns
            sharpe_reward * 2.0 +          # Risk-adjusted returns
            drawdown_penalty +             # Penalize large drawdowns
            cost_penalty * 5.0             # Penalize excessive trading
        )
        
        return reward * self.reward_scaling
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns history"""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns_array = np.array(list(self.returns_history))
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
        return np.clip(sharpe, -5.0, 5.0)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak"""
        if len(self.equity_history) < 2:
            return 0.0
        
        equity_array = np.array(list(self.equity_history))
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / (peak + 1e-8)
        max_dd = np.max(drawdown)
        
        return float(np.clip(max_dd, 0.0, 1.0))
    
    def render(self):
        """Render environment state"""
        if len(self.equity_history) == 0:
            return
        
        current_value = self.equity_history[-1]
        total_return = (current_value - self.initial_balance) / self.initial_balance
        
        print(f"\n{'='*60}")
        print(f"Step: {self.step_idx} | Portfolio Value: ${current_value:,.2f}")
        print(f"Return: {total_return:+.2%} | Cash: ${self.cash:,.2f}")
        print(f"{'='*60}")
        
        # Show holdings
        prices = self._get_current_prices()
        print("\nHoldings:")
        for i, ticker in enumerate(self.stock_tickers):
            shares = self.shares[ticker]
            value = shares * prices[i]
            weight = self.portfolio_weights[i]
            print(f"  {ticker}: {shares:.2f} shares @ ${prices[i]:.2f} = ${value:,.2f} ({weight:.1%})")
        
        cash_weight = self.portfolio_weights[-1]
        print(f"  CASH: ${self.cash:,.2f} ({cash_weight:.1%})")
        print(f"{'='*60}\n")
