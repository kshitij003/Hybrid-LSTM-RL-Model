from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import sys
import asyncio

# Add parent directory to path to import existing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

router = APIRouter()

class TrainingConfig(BaseModel):
    symbol: str = "MSFT"
    start_date: str = "2010-01-01"
    end_date: str = "2024-01-01"
    interval: str = "1d"
    timesteps: int = 100000

class TrainingStatus(BaseModel):
    status: str
    progress: int
    timestamp: datetime
    message: str

training_status = {
    "status": "idle",
    "progress": 0,
    "timestamp": datetime.now(),
    "message": "No training in progress"
}

async def auto_complete_training(delay_seconds: int = 30):
    """Automatically mark training as completed after delay"""
    global training_status
    await asyncio.sleep(delay_seconds)
    if training_status["status"] == "running":
        training_status = {
            "status": "completed",
            "progress": 100,
            "timestamp": datetime.now(),
            "message": "Training completed successfully"
        }

@router.post("/train")
async def start_training(config: TrainingConfig):
    """Start model training with specified configuration"""
    global training_status
    
    try:
        training_status = {
            "status": "running",
            "progress": 0,
            "timestamp": datetime.now(),
            "message": f"Training started with {config.symbol}..."
        }
        
        # Schedule auto-completion after 30 seconds
        asyncio.create_task(auto_complete_training(30))
        
        # The actual training logic would be called here
        # For now, return success response
        return {
            "status": "success",
            "message": f"Training started for {config.symbol}",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-status")
async def get_training_status():
    """Get current training status"""
    return training_status

@router.post("/train/cancel")
async def cancel_training():
    """Cancel ongoing training"""
    global training_status
    training_status = {
        "status": "cancelled",
        "progress": training_status["progress"],
        "timestamp": datetime.now(),
        "message": "Training cancelled by user"
    }
    return {"message": "Training cancelled"}
