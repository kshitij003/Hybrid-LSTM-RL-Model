from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Check if API is running"""
    return {
        "status": "healthy",
        "message": "API is running"
    }

@router.get("/status")
async def get_status():
    """Get system status"""
    return {
        "backend": "running",
        "model_loaded": False,
        "last_training": None
    }
