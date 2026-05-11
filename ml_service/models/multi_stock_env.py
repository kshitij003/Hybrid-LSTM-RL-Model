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
        reward_scaling: float = 1.0,
        patience_window: int = 3,        # consecutive down-days before sell is rewarded
        min_dip_threshold: float = 0.03  # minimum drawdown from peak to trigger cash bonus
    ):
        """
        Initialize multi-stock portfolio environment.

        Args:
            stock_dataframes:  Dict of {ticker: DataFrame} with OHLCV + features
            feature_columns:   List of feature column names
            initial_balance:   Starting cash balance
            transaction_cost:  Trading cost as fraction of trade value
            lstm_predictor:    Optional LSTM predictor for enhanced observations
            sequence_length:   Lookback period for LSTM
            reward_scaling:    Scale factor for rewards
            patience_window:   # of consecutive negative-return days the agent must
                               observe before a sell is rewarded (prevents panic-sells
                               on single red candles)
            min_dip_threshold: Minimum drawdown from recent peak (fraction) before
                               the cash-safe-haven bonus activates (filters noise < 3%)
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
        self.patience_window   = patience_window
        self.min_dip_threshold = min_dip_threshold
        
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
        self.equity_history   = deque(maxlen=100)
        self.returns_history  = deque(maxlen=20)
        # Patience tracking
        self.consecutive_down_days = 0   # counts consecutive negative-return steps
        self.peak_value            = float(initial_balance)  # rolling portfolio peak
        
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

        # Reset patience trackers
        self.consecutive_down_days = 0
        self.peak_value            = self.initial_balance

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
        """Get current raw close prices for all stocks"""
        # Look for Raw_Close first (Dual-Price strategy), fallback to Close
        prices = []
        for ticker in self.stock_tickers:
            df = self.dfs[ticker]
            col = "Raw_Close" if "Raw_Close" in df.columns else "Close"
            prices.append(float(df[col].iloc[self.step_idx]))
            
        prices = np.array(prices, dtype=np.float32)
        
        # Safer price check
        if np.any(np.isnan(prices)) or np.any(prices <= 0):
            prices = np.nan_to_num(prices, nan=1.0)
            prices[prices <= 0] = 1.0
            
        return prices
    
    def _get_prices_at_step(self, step: int) -> np.ndarray:
        """Get raw prices at specific step"""
        prices = []
        for ticker in self.stock_tickers:
            df = self.dfs[ticker]
            col = "Raw_Close" if "Raw_Close" in df.columns else "Close"
            prices.append(float(df[col].iloc[step]))
            
        prices = np.array(prices, dtype=np.float32)
        return np.maximum(prices, 1.0)
    
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
        
        if self.step_idx % 10 == 0:
             print(f"   [DEBUG] Step {self.step_idx}: Cash=₹{self.cash:.0f}, Total=₹{current_value:.0f}")

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
        Rebalance portfolio to match target weights using REALISTIC constraints:
        - Integer shares only
        - Strict cash enforcement
        """
        current_value = self._calculate_total_value()
        prices = self._get_current_prices()
        total_cost = 0.0
        
        # 1. Sell stocks first to free up cash
        for i, ticker in enumerate(self.stock_tickers):
            target_value = target_weights[i] * current_value
            current_shares = self.shares[ticker]
            current_stock_value = current_shares * prices[i]
            
            if current_stock_value > target_value:
                # Need to sell
                value_to_sell = current_stock_value - target_value
                shares_to_sell = int(np.floor(value_to_sell / prices[i]))
                
                if shares_to_sell > 0:
                    shares_to_sell = min(shares_to_sell, current_shares)
                    trade_value = shares_to_sell * prices[i]
                    cost = trade_value * self.transaction_cost
                    self.shares[ticker] -= shares_to_sell
                    self.cash += (trade_value - cost)
                    total_cost += cost

        # 2. Buy stocks with available cash
        # We sort by weight to prioritize stocks the agent wants most
        buy_indices = np.argsort(target_weights[:-1])[::-1]
        
        for i in buy_indices:
            ticker = self.stock_tickers[i]
            target_value = target_weights[i] * current_value
            current_stock_value = self.shares[ticker] * prices[i]
            
            if target_value > current_stock_value:
                # Need to buy
                available_for_stock = target_value - current_stock_value
                # Can only buy with actual cash on hand
                available_for_stock = min(available_for_stock, self.cash * 0.99) # leave room for costs
                
                shares_to_buy = int(np.floor(available_for_stock / prices[i]))
                
                if shares_to_buy > 0:
                    trade_value = shares_to_buy * prices[i]
                    cost = trade_value * self.transaction_cost
                    
                    if self.cash >= (trade_value + cost):
                        self.shares[ticker] += shares_to_buy
                        self.cash -= (trade_value + cost)
                        total_cost += cost
        
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
        Patience-aware Profit-first Reward Function.

        Key design principles:
        ─────────────────────
        1. PROFIT IS DOMINANT — return_pct weight is highest so the agent
           always prefers making money over hiding in cash.
        2. PATIENCE WINDOW — The agent must observe `patience_window`
           consecutive down-days before a sell is rewarded. On 1–2 bad days
           a hold-bonus dampens the urge to panic-sell noise.
        3. DIP THRESHOLD — The cash safe-haven bonus only activates when
           drawdown from the recent peak exceeds `min_dip_threshold` (3%).
           Tiny intraday dips are completely ignored.
        4. HOLDING BONUS — The agent earns a small reward each step it holds
           a profitable position, directly competing with the cash incentive.
        5. LIGHTER DRAWDOWN PENALTY — Was ×5.0, now ×1.5 so major crashes
           are still penalised without triggering panic on small corrections.
        6. SORTINO TERM REDUCED — Was ×2.0, now ×0.8 to prevent short-term
           volatility from swamping the profit signal.
        """
        value_change = current_value - prev_value
        return_pct   = value_change / prev_value if prev_value > 0 else 0.0

        # ── Update patience counter ────────────────────────────────────────
        if return_pct < 0:
            self.consecutive_down_days += 1
        else:
            self.consecutive_down_days = 0   # reset on any positive day

        # ── Update rolling peak ───────────────────────────────────────────
        if current_value > self.peak_value:
            self.peak_value = current_value

        # ── 1. Primary profit signal (dominant term) ───────────────────────
        profit_reward = return_pct * 15.0

        # ── 2. Holding bonus — reward staying invested while in profit ─────
        #    Small per-step bonus that competes with the cash incentive
        if current_value > self.initial_balance and self.portfolio_weights[-1] < 0.5:
            holding_bonus = 0.002   # +0.2% per step for being invested profitably
        else:
            holding_bonus = 0.0

        # ── 3. Patience-gated cash bonus ──────────────────────────────────
        #    Only reward going to cash when BOTH conditions are met:
        #      a) drawdown from peak exceeds min_dip_threshold (3%)
        #      b) agent has been patient for >= patience_window down-days
        drawdown_from_peak = (self.peak_value - current_value) / (self.peak_value + 1e-8)
        cash_weight        = self.portfolio_weights[-1]

        major_dip      = drawdown_from_peak >= self.min_dip_threshold
        patient_enough = self.consecutive_down_days >= self.patience_window

        if major_dip and patient_enough:
            # Real crash — reward holding cash
            cash_bonus = cash_weight * drawdown_from_peak * 3.0
        elif return_pct < 0 and not major_dip:
            # Minor dip (< 3%) — small penalty for unnecessary panic-selling
            cash_bonus = -cash_weight * abs(return_pct) * 2.0
        else:
            # Rising market — penalise sitting on cash (opportunity cost)
            cash_bonus = -cash_weight * return_pct * 1.0

        # ── 4. Sortino risk term (reduced weight) ─────────────────────────
        if len(self.returns_history) >= 5:
            returns_array    = np.array(list(self.returns_history))
            mean_return      = np.mean(returns_array)
            negative_returns = returns_array[returns_array < 0]
            downside_std     = np.std(negative_returns) + 1e-6 if len(negative_returns) > 0 else 1e-6
            sortino_reward   = mean_return / downside_std
        else:
            sortino_reward = 0.0

        # ── 5. Drawdown penalty (lighter — was ×5.0) ──────────────────────
        max_drawdown     = self._calculate_max_drawdown()
        drawdown_penalty = -max_drawdown * 1.5

        # ── 6. Transaction cost penalty (discourages churning) ────────────
        cost_penalty = -transaction_costs / prev_value * 2.0 if prev_value > 0 else 0.0

        # ── Combined reward ───────────────────────────────────────────────
        reward = (
            profit_reward    +   # 15× return  — dominant signal
            holding_bonus    +   # +0.002/step  — stay invested while profitable
            cash_bonus       +   # context-aware safe-haven logic (3% dip gate)
            sortino_reward * 0.8 +  # light risk-adjustment
            drawdown_penalty +   # major-crash deterrent (was ×5, now ×1.5)
            cost_penalty         # anti-churn
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
