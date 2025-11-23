import gymnasium as gym
from gymnasium import spaces
import numpy as np


class StockTradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        feature_cols,
        initial_balance=100000,
        transaction_fee=0.001,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols

        self.data = df[feature_cols].values.astype(np.float32)
        self.close = df["Close"].values.astype(np.float32)

        self.n_steps = len(df)

        # --------- Portfolio State ---------
        self.initial_balance = initial_balance
        self.equity = initial_balance     # TOTAL portfolio equity
        self.position = 0.0               # allocation -1 (short) to +1 (long)

        self.prev_equity = self.equity
        self.transaction_fee = transaction_fee
        self.current_step = 0

        # --------- Action: allocation [-1, 1] ---------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # --------- Observation: features + position + normalized_equity ---------
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(feature_cols) + 2,),
            dtype=np.float32
        )

        # --------- Metrics ---------
        self.equity_curve = []
        self.returns = []
        self.position_history = []
        self.win_trades = 0
        self.loss_trades = 0

    # --------------------------------------------------
    def _get_obs(self):
        feat = self.data[self.current_step]
        norm_equity = self.equity / self.initial_balance
        return np.concatenate([feat, [self.position, norm_equity]]).astype(np.float32)

    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
        self.position = 0.0
        self.current_step = 0

        self.equity_curve = []
        self.returns = []
        self.position_history = []
        self.win_trades = 0
        self.loss_trades = 0

        return self._get_obs(), {}

    # --------------------------------------------------
    def step(self, action):

        action = float(action[0])
        action = np.clip(action, -1, 1)

        prev_position = self.position
        prev_price = self.close[self.current_step]

        # Advance timestep
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        price_now = self.close[self.current_step]

        # Avoid division by zero
        if prev_price <= 0:
            prev_price = 1e-6

        # ---------- Apply new allocation ----------
        self.position = action

        # ---------- Trading Fee ----------
        fee = abs(self.position - prev_position) * self.transaction_fee * self.equity
        self.equity -= fee

        # ---------- Mark-to-market PnL ----------
        price_change = (price_now - prev_price) / prev_price
        pnl = self.equity * prev_position * price_change

        if not np.isnan(pnl):
            self.equity += pnl
        else:
            pnl = 0

        # ---------- Reward ----------
        if self.prev_equity != 0:
            reward = (self.equity - self.prev_equity) / self.prev_equity
        else:
            reward = 0.0

        self.prev_equity = max(self.equity, 1e-6)

        # ---------- Metrics ----------
        self.position_history.append(self.position)
        self.equity_curve.append(self.equity)
        self.returns.append(reward)

        if pnl > 0:
            self.win_trades += 1
        elif pnl < 0:
            self.loss_trades += 1

        info = {
            "equity": self.equity,
            "equity_curve": self.equity_curve,
            "returns": self.returns,
            "win_rate": self._winrate(),
            "max_drawdown": self._max_drawdown(),
            "sharpe": self._sharpe(),
            "sortino": self._sortino(),
            "position_history": self.position_history,
        }

        return self._get_obs(), float(reward), done, False, info

    # ==================================================
    # METRICS
    # ==================================================

    def _winrate(self):
        total = self.win_trades + self.loss_trades
        if total == 0:
            return 0
        return self.win_trades / total

    def _max_drawdown(self):
        if len(self.equity_curve) < 2:
            return 0
        equity = np.array(self.equity_curve)
        peaks = np.maximum.accumulate(equity)
        dd = (equity - peaks) / peaks
        return dd.min()

    def _sharpe(self):
        if len(self.returns) < 2:
            return 0
        r = np.array(self.returns)
        if r.std() == 0:
            return 0
        return np.sqrt(252) * r.mean() / r.std()

    def _sortino(self):
        if len(self.returns) < 2:
            return 0
        r = np.array(self.returns)
        downside = r[r < 0]
        if downside.std() == 0:
            return 0
        return np.sqrt(252) * r.mean() / downside.std()

    # --------------------------------------------------
    def render(self):
        print(
            f"Step {self.current_step} | Equity: {self.equity:.2f} | Pos: {self.position:.2f}"
        )

    def close(self):
        pass
