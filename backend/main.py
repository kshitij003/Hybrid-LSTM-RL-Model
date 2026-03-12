from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import training, backtesting, prediction, health

app = FastAPI(
    title="Hybrid LSTM-RL Trading API",
    description="API for training and backtesting hybrid LSTM-RL trading model",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(training.router, prefix="/api", tags=["Training"])
app.include_router(backtesting.router, prefix="/api", tags=["Backtesting"])
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])

@app.get("/")
async def root():
    return {
        "message": "Hybrid LSTM-RL Trading API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
