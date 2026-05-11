"""
Portfolio Preferences API Blueprint
Handles user stock selection and triggers custom LSTM+PPO quick-update training.
"""

from flask import Blueprint, request, jsonify
import os
import json
import threading
import logging
from datetime import datetime

portfolio_bp = Blueprint('portfolio', __name__)
logger = logging.getLogger(__name__)

PREFERENCES_FILE = 'models/saved_models/user_preferences.json'
MODEL_DIR = 'models/saved_models'

DEFAULT_STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

NSE_STOCKS = [
    {"symbol": "RELIANCE.NS",    "name": "Reliance Industries",         "sector": "Energy"},
    {"symbol": "TCS.NS",         "name": "Tata Consultancy Services",    "sector": "IT"},
    {"symbol": "HDFCBANK.NS",    "name": "HDFC Bank",                    "sector": "Banking"},
    {"symbol": "INFY.NS",        "name": "Infosys",                      "sector": "IT"},
    {"symbol": "ICICIBANK.NS",   "name": "ICICI Bank",                   "sector": "Banking"},
    {"symbol": "WIPRO.NS",       "name": "Wipro",                        "sector": "IT"},
    {"symbol": "HINDUNILVR.NS",  "name": "Hindustan Unilever",           "sector": "FMCG"},
    {"symbol": "BAJFINANCE.NS",  "name": "Bajaj Finance",                "sector": "Finance"},
    {"symbol": "KOTAKBANK.NS",   "name": "Kotak Mahindra Bank",          "sector": "Banking"},
    {"symbol": "AXISBANK.NS",    "name": "Axis Bank",                    "sector": "Banking"},
    {"symbol": "LT.NS",          "name": "Larsen & Toubro",              "sector": "Infrastructure"},
    {"symbol": "MARUTI.NS",      "name": "Maruti Suzuki",                "sector": "Auto"},
    {"symbol": "SUNPHARMA.NS",   "name": "Sun Pharmaceutical",           "sector": "Pharma"},
    {"symbol": "TITAN.NS",       "name": "Titan Company",                "sector": "Consumer"},
    {"symbol": "ADANIENT.NS",    "name": "Adani Enterprises",            "sector": "Conglomerate"},
    {"symbol": "ADANIPORTS.NS",  "name": "Adani Ports",                  "sector": "Infrastructure"},
    {"symbol": "TATAMOTORS.NS",  "name": "Tata Motors",                  "sector": "Auto"},
    {"symbol": "TATASTEEL.NS",   "name": "Tata Steel",                   "sector": "Metals"},
    {"symbol": "ONGC.NS",        "name": "ONGC",                         "sector": "Energy"},
    {"symbol": "NTPC.NS",        "name": "NTPC",                         "sector": "Power"},
    {"symbol": "POWERGRID.NS",   "name": "Power Grid Corp",              "sector": "Power"},
    {"symbol": "COALINDIA.NS",   "name": "Coal India",                   "sector": "Mining"},
    {"symbol": "JSWSTEEL.NS",    "name": "JSW Steel",                    "sector": "Metals"},
    {"symbol": "BHARTIARTL.NS",  "name": "Bharti Airtel",                "sector": "Telecom"},
    {"symbol": "SBIN.NS",        "name": "State Bank of India",          "sector": "Banking"},
    {"symbol": "M&M.NS",         "name": "Mahindra & Mahindra",          "sector": "Auto"},
    {"symbol": "HEROMOTOCO.NS",  "name": "Hero MotoCorp",                "sector": "Auto"},
    {"symbol": "DRREDDY.NS",     "name": "Dr. Reddy's Laboratories",     "sector": "Pharma"},
    {"symbol": "ULTRACEMCO.NS",  "name": "UltraTech Cement",             "sector": "Cement"},
    {"symbol": "GRASIM.NS",      "name": "Grasim Industries",            "sector": "Diversified"},
    {"symbol": "BAJAJ-AUTO.NS",  "name": "Bajaj Auto",                   "sector": "Auto"},
    {"symbol": "TECHM.NS",       "name": "Tech Mahindra",                "sector": "IT"},
    {"symbol": "HCLTECH.NS",     "name": "HCL Technologies",             "sector": "IT"},
    {"symbol": "ASIANPAINT.NS",  "name": "Asian Paints",                 "sector": "Consumer"},
    {"symbol": "NESTLEIND.NS",   "name": "Nestle India",                 "sector": "FMCG"},
]


# ── GET preferences ───────────────────────────────────────────────────────────

@portfolio_bp.route('/preferences', methods=['GET'])
def get_preferences():
    """Return current user stock preferences."""
    return jsonify(_load_preferences()), 200


# ── POST preferences ──────────────────────────────────────────────────────────

@portfolio_bp.route('/preferences', methods=['POST'])
def save_preferences():
    """Save user stock preferences (1–10 NSE stocks)."""
    data = request.json
    if not data or 'stocks' not in data:
        return jsonify({"error": {"code": "INVALID_REQUEST", "message": "Missing 'stocks' field"}}), 400

    stocks = data['stocks']
    if not (1 <= len(stocks) <= 10):
        return jsonify({"error": {"code": "CONSTRAINT_VIOLATION",
                                  "message": "Must select between 1 and 10 stocks"}}), 400

    prefs = {
        "stocks":              stocks,
        "rebalanceFrequency":  data.get("rebalanceFrequency", "Monthly"),
        "initialValueType":    data.get("initialValueType", "Fixed"),
        "initialValueUnit":    data.get("initialValueUnit", "Percent"),
        "updatedAt":           datetime.now().isoformat(),
    }
    _save_preferences(prefs)
    return jsonify({"status": "SAVED", "preferences": prefs}), 200


# ── Supported stocks search ───────────────────────────────────────────────────

@portfolio_bp.route('/supported-stocks', methods=['GET'])
def get_supported_stocks():
    """Return list of supported NSE stocks, optionally filtered by ?q=query."""
    q = request.args.get('q', '').upper().strip()
    stocks = (
        [s for s in NSE_STOCKS if q in s['symbol'] or q in s['name'].upper()]
        if q else NSE_STOCKS
    )
    return jsonify({"stocks": stocks, "total": len(stocks)}), 200


# ── Trigger custom portfolio training ─────────────────────────────────────────

@portfolio_bp.route('/train-custom', methods=['POST'])
def train_custom_portfolio():
    """
    Quick-update: train LSTM for new stocks only, then fine-tune existing PPO.
    Falls back to full retrain if no base PPO exists.

    Request Body (all optional — falls back to saved preferences):
    {
        "stocks": ["RELIANCE.NS", "TCS.NS", ...],
        "config": { "ppoTimesteps": 50000, "lstmEpochs": 10, ... }
    }
    """
    from api.training import training_jobs

    data    = request.json or {}
    prefs   = _load_preferences()
    stocks  = data.get('stocks', prefs.get('stocks', DEFAULT_STOCKS))
    config  = data.get('config', {})

    training_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ppo_path    = os.path.join(MODEL_DIR, 'ppo_multi_stock.zip')

    training_jobs[training_id] = {
        "trainingId": training_id,
        "type":       "CUSTOM_PORTFOLIO",
        "status":     "QUEUED",
        "stocks":     stocks,
        "config":     config,
        "progress":   {"stage": "INITIALIZING", "percentComplete": 0.0},
        "startedAt":  datetime.now().isoformat(),
        "updatedAt":  datetime.now().isoformat(),
    }

    if os.path.exists(ppo_path):
        target = _run_custom_quick_update
        mode   = "QUICK_UPDATE"
        eta    = "15-30 minutes"
    else:
        target = _run_custom_full_train
        mode   = "FULL_TRAIN"
        eta    = "2-3 hours"

    threading.Thread(target=target, args=(training_id, stocks, config), daemon=True).start()

    return jsonify({
        "trainingId":    training_id,
        "status":        "STARTED",
        "mode":          mode,
        "stocks":        stocks,
        "estimatedTime": eta,
        "message":       f"Custom portfolio training started for {len(stocks)} stocks ({mode})",
    }), 202


# ── Background workers ────────────────────────────────────────────────────────

def _run_custom_quick_update(training_id: str, stocks: list, config: dict):
    """
    Quick-update flow:
      1. Download market data for all selected stocks.
      2. Train LSTM only for stocks that don't have a saved .pth file.
      3. Load all LSTM models for the selected stocks.
      4. Fine-tune existing PPO on a fresh env built from those stocks only.
      5. Save & hot-reload.
    """
    from api.training import training_jobs
    job = training_jobs[training_id]

    try:
        job['status']    = 'IN_PROGRESS'
        job['updatedAt'] = datetime.now().isoformat()

        from data.data_handler          import DataHandler
        from data.feature_engineer      import FeatureEngineer
        from models.multi_stock_lstm    import MultiStockLSTMPredictor, train_multi_stock_lstm
        from models.multi_stock_env     import MultiStockPortfolioEnv
        from stable_baselines3          import PPO
        from stable_baselines3.common.callbacks import BaseCallback

        ppo_path        = os.path.join(MODEL_DIR, 'ppo_multi_stock')
        start_date      = config.get('startDate',       '2023-01-01')
        end_date        = config.get('endDate',         '2025-01-01')
        seq_len         = config.get('sequenceLength',  30)
        init_balance    = config.get('initialBalance',  100000)
        ppo_timesteps   = config.get('ppoTimesteps',    50000)
        lstm_epochs     = config.get('lstmEpochs',      10)

        def _upd(stage, pct):
            job['progress'] = {'stage': stage, 'percentComplete': round(pct, 1)}
            job['updatedAt'] = datetime.now().isoformat()

        # ── 1. Download data ──────────────────────────────────────────────────
        _upd('DATA_DOWNLOAD', 5.0)
        stock_dfs    = {}
        feature_cols = None

        for i, ticker in enumerate(stocks):
            handler = DataHandler(
                symbols=[ticker], start_date=start_date, end_date=end_date,
                cache_file=f"data/cache/{ticker}_custom.csv"
            )
            df = handler.download_data()
            fe = FeatureEngineer()
            sentiment = fe.fetch_and_score_sentiment(ticker=ticker, df_index=df.index, days_back=30)
            df_feat, fcols = fe.get_feature_df(df, sentiment_scores=sentiment)
            feature_cols = fcols
            stock_dfs[ticker] = df_feat
            _upd('DATA_DOWNLOAD', 5.0 + (i + 1) / len(stocks) * 15.0)

        # ── 2. Train LSTM for new stocks only ─────────────────────────────────
        _upd('LSTM_TRAINING', 20.0)
        needs_lstm = [t for t in stocks
                      if not os.path.exists(os.path.join(MODEL_DIR, f'lstm_{t}.pth'))]

        if needs_lstm:
            logger.info(f"[{training_id}] Training new LSTMs for: {needs_lstm}")
            train_multi_stock_lstm(
                stock_dataframes={t: stock_dfs[t] for t in needs_lstm},
                feature_columns=feature_cols,
                save_dir=MODEL_DIR,
                sequence_length=seq_len,
                epochs=lstm_epochs,
            )

        # ── 3. Load all LSTM models for selected stocks ───────────────────────
        _upd('LOADING_LSTM', 38.0)
        lstm_paths = {t: os.path.join(MODEL_DIR, f'lstm_{t}.pth')
                      for t in stocks
                      if os.path.exists(os.path.join(MODEL_DIR, f'lstm_{t}.pth'))}

        lstm_predictor = MultiStockLSTMPredictor(
            stock_model_paths=lstm_paths,
            input_dim=len(feature_cols),
            hidden_dim=50,
        )

        # ── 4. Build env & fine-tune PPO ──────────────────────────────────────
        _upd('BUILDING_ENV', 42.0)
        env = MultiStockPortfolioEnv(
            stock_dataframes=stock_dfs,
            feature_columns=feature_cols,
            lstm_predictor=lstm_predictor,
            initial_balance=init_balance,
            sequence_length=seq_len,
        )

        _upd('PPO_FINETUNING', 45.0)
        model = PPO.load(ppo_path, env=env)

        class _ProgressCB(BaseCallback):
            def __init__(self, job_dict, total):
                super().__init__()
                self._job  = job_dict
                self._total = total

            def _on_step(self):
                pct = 45.0 + (self.num_timesteps / self._total) * 50.0
                self._job['progress'] = {
                    'stage':           'PPO_FINETUNING',
                    'percentComplete':  round(pct, 1),
                    'currentTimestep':  self.num_timesteps,
                    'totalTimesteps':   self._total,
                }
                self._job['updatedAt'] = datetime.now().isoformat()
                return True

        model.learn(
            total_timesteps=ppo_timesteps,
            reset_num_timesteps=False,
            callback=_ProgressCB(job, ppo_timesteps),
        )

        # ── 5. Save & hot-reload ──────────────────────────────────────────────
        tag          = "_".join(s.replace(".NS", "") for s in stocks[:3])
        custom_path  = os.path.join(MODEL_DIR, f'ppo_custom_{tag}')
        model.save(custom_path)
        model.save(ppo_path)          # overwrite active model

        _save_preferences({
            "stocks":            stocks,
            "updatedAt":         datetime.now().isoformat(),
            "lastTrainedModel":  custom_path + ".zip",
        })

        job.update({
            'status':      'COMPLETED',
            'progress':    {'stage': 'FINISHED', 'percentComplete': 100.0},
            'completedAt': datetime.now().isoformat(),
            'updatedAt':   datetime.now().isoformat(),
            'results': {
                'modelPath': custom_path + '.zip',
                'stocks':    stocks,
                'type':      'CUSTOM_QUICK_UPDATE',
            },
        })

        from api.inference import load_active_model
        load_active_model()
        logger.info(f"[{training_id}]  Custom portfolio training complete.")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        job.update({'status': 'FAILED', 'error': str(exc), 'updatedAt': datetime.now().isoformat()})
        logger.error(f"[{training_id}]  Custom training failed: {exc}")


def _run_custom_full_train(training_id: str, stocks: list, config: dict):
    """Full retrain when no base PPO model exists."""
    from api.training import training_jobs, run_training_job
    start_date = config.get('startDate', '2020-01-01')
    end_date   = config.get('endDate',   '2025-01-01')
    save_path  = os.path.join(MODEL_DIR, f'ppo_custom_{training_id}')
    run_training_job(training_id, stocks, start_date, end_date, config, save_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_preferences() -> dict:
    os.makedirs(os.path.dirname(PREFERENCES_FILE), exist_ok=True)
    if os.path.exists(PREFERENCES_FILE):
        with open(PREFERENCES_FILE, 'r') as f:
            return json.load(f)
    return {
        "stocks":             DEFAULT_STOCKS,
        "rebalanceFrequency": "Monthly",
        "initialValueType":   "Fixed",
        "initialValueUnit":   "Percent",
    }


def _save_preferences(prefs: dict):
    os.makedirs(os.path.dirname(PREFERENCES_FILE), exist_ok=True)
    with open(PREFERENCES_FILE, 'w') as f:
        json.dump(prefs, f, indent=2)
