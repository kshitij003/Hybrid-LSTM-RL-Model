from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime

router = APIRouter()

class BacktestConfig(BaseModel):
    symbol: str = "MSFT"
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    model_path: Optional[str] = None

class BacktestResults(BaseModel):
    final_equity: float
    total_return: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    avg_trade_return: float

@router.post("/backtest")
async def run_backtest(config: BacktestConfig):
    """Run backtest on model with historical data"""
    try:
        # Mock backtest results - will connect to actual backtest.py
        results = {
            "status": "success",
            "timestamp": datetime.now(),
            "config": config.dict(),
            "results": {
                "final_equity": 150000.50,
                "total_return": 50000.50,
                "win_rate": 0.62,
                "max_drawdown": 5000.25,
                "sharpe_ratio": 1.45,
                "total_trades": 156,
                "avg_trade_return": 320.52
            }
        }
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest/history")
async def get_backtest_history():
    """Get previous backtest results"""
    return {
        "backtests": [
            {
                "id": 1,
                "timestamp": "2024-01-15",
                "symbol": "MSFT",
                "final_equity": 150000.50,
                "total_return": 50000.50,
                "win_rate": 0.62
            },
            {
                "id": 2,
                "timestamp": "2024-01-10",
                "symbol": "AAPL",
                "final_equity": 145000.25,
                "total_return": 45000.25,
                "win_rate": 0.58
            }
        ]
    }

@router.get("/backtest/{backtest_id}/details")
async def get_backtest_details(backtest_id: int):
    """Get detailed results of a specific backtest"""
    return {
        "id": backtest_id,
        "results": {
            "portfolio_curve": [100000, 101500, 103200, 98500, 105000],
            "actions": [0, 1, 0, 2, 1],
            "prices": [100.0, 101.5, 103.2, 98.5, 105.0]
        }
    }
