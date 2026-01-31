import gymnasium as gym
import numpy as np


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, feature_columns, initial_balance=10_000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = float(initial_balance)

        # --- PPO-friendly action space ---
        # 0 = hold, 1 = buy, 2 = sell
        self.action_space = gym.spaces.Discrete(3)

        # --- observation space ---
        # features (normalized) + balance% + inventory%
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(feature_columns) + 2,),
            dtype=np.float32,
        )

    # ------------------------------------
    # RESET
    # ------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_idx = 0
        self.balance = self.initial_balance
        self.inventory = 0
        self.prev_equity = self.initial_balance

        return self._get_obs(), {}

    # ------------------------------------
    # OBSERVATION
    # ------------------------------------
    def _get_obs(self):
        features = self.df.loc[self.step_idx, self.feature_columns].values.astype(np.float32)

        balance_pct = np.array([self.balance / self.initial_balance], dtype=np.float32)
        inv_pct = np.array([self.inventory / 1000.0], dtype=np.float32)  # scaled inventory

        # clip for safety
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        balance_pct = np.clip(balance_pct, 0, 10)
        inv_pct = np.clip(inv_pct, 0, 10)

        return np.concatenate([features, balance_pct, inv_pct])

    # ------------------------------------
    # STEP
    # ------------------------------------
    def step(self, action):
        price = float(self.df["Close"].iloc[self.step_idx])
        price = max(price, 0.001)

        # --------- Execute Action ---------
        if action == 1:  # BUY
            if self.balance >= price:
                shares = int(self.balance // price)
                shares = min(shares, 1000)  # cap to prevent infinite growth
                if shares > 0:
                    self.balance -= shares * price
                    self.inventory += shares

        elif action == 2:  # SELL
            if self.inventory > 0:
                self.balance += self.inventory * price
                self.inventory = 0

        # --------- Reward: equity change ---------
        # ----- ADVANCED REWARD SYSTEM -----

# 1. Equity
        total_equity = self.balance + self.inventory * price
        profit_change = total_equity - self.prev_equity

        # 2. Track equity history for sharpe + volatility
        if not hasattr(self, "equity_curve"):
            self.equity_curve = []

        self.equity_curve.append(total_equity)

        # 3. Return
        ret = profit_change / max(self.prev_equity, 1e-6)

        # 4. Sharpe-like term (1-step approximation)
        sharpe_reward = ret / (abs(ret) + 1e-6)

        # 5. Volatility penalty (smoothness)
        if len(self.equity_curve) > 3:
            recent = np.array(self.equity_curve[-5:])
            vol = np.std(np.diff(recent))
        else:
            vol = 0.0

        vol_penalty = -vol * 0.1    # tuneable

        # 6. Drawdown penalty
        if len(self.equity_curve) > 5:
            peak = np.max(self.equity_curve)
            dd = (peak - total_equity) / peak
        else:
            dd = 0.0

        drawdown_penalty = -dd * 0.2  # tuneable

        # 7. Final reward
        reward = (
            profit_change * 0.4 +       # profit is still important
            sharpe_reward * 1.0 +       # smooth profit encouraged
            vol_penalty +               # reduce volatility
            drawdown_penalty            # avoid big losses
        )

        # update equity
        self.prev_equity = total_equity


        # --------- Step Forward ---------
        self.step_idx += 1
        terminated = self.step_idx >= len(self.df) - 1

        return self._get_obs(), float(reward), terminated, False, {}

    # ------------------------------------
    def render(self):
        print(
            f"Step: {self.step_idx} | Price: {self.df['Close'].iloc[self.step_idx]:.2f} | "
            f"Balance: {self.balance:.2f} | Inventory: {self.inventory} | "
            f"Equity: {self.prev_equity:.2f}"
        )
