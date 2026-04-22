import os
import shutil
import time
import requests
import subprocess
import sys
import threading

ML_SERVICE_DIR = r"c:\Users\Admin\Desktop\Hybrid-LSTM-RL-Project\ml_service"
ROOT_MODELS_DIR = r"c:\Users\Admin\Desktop\Hybrid-LSTM-RL-Project\models"
ML_MODELS_DIR = os.path.join(ML_SERVICE_DIR, "models", "saved_models")
CACHE_DIR = os.path.join(ML_SERVICE_DIR, "data", "cache")

def log_streamer(pipe, prefix):
    """Streams lines from a pipe with a prefix."""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                print(f"{prefix} {line.strip()}")
    except Exception as e:
        print(f"Error in log streamer: {e}")
    finally:
        pipe.close()

print("Clearing old models and cache...")
for folder in [ROOT_MODELS_DIR, ML_MODELS_DIR, CACHE_DIR]:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
print("Old models and cache cleared.")

print("Starting ml_service...")
# Start Flask app in background
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["DEBUG"] = "true"  # Enable debug mode for more logs

process = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd=ML_SERVICE_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,
    encoding="utf-8",
    bufsize=1 # Line buffered
)

# Start log streaming thread
logger_thread = threading.Thread(
    target=log_streamer, 
    args=(process.stdout, "[ML_SERVICE]"),
    daemon=True
)
logger_thread.start()

# Wait for server to boot
print("Waiting 15 seconds for server to start...")
time.sleep(15)

if process.poll() is not None:
    print(f"Server crashed with exit code {process.returncode}!")
    sys.exit(1)

print("Triggering full training on Indian stocks...")
try:
    response = requests.post(
        "http://localhost:8000/api/train/multi-stock",
        json={
            "stocks": [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"
            ],
            "startDate": "2023-01-01",  # ~480 trading days ago
            "endDate": "2024-12-31",
            "config": {
                "lstmEpochs": 20,       # Full LSTM training
                "ppoTimesteps": 500000, # 500k RL steps
                "initialBalance": 100000
            }
        }
    )
    response.raise_for_status()
    data = response.json()
    job_id = data.get("trainingId")
    print(f"Training started. Job ID: {job_id}")

    print("Polling status...")
    while True:
        status_resp = requests.get(f"http://localhost:8000/api/train/status/{job_id}")
        status_data = status_resp.json()
        stage = status_data.get("progress", {}).get("stage", "UNKNOWN")
        pct = status_data.get("progress", {}).get("percentComplete", 0)
        
        print(f"   [{pct:.1f}%] Stage: {stage}")
        
        if stage == "COMPLETED":
            print("Training completed successfully!")
            break
        elif stage == "FAILED":
            print("Training failed!")
            print(status_data)
            break
            
        time.sleep(10)

except Exception as e:
    print(f"Error communicating with ml_service: {e}")
    if process.poll() is not None:
        print(f"Server process terminated unexpectedly with code {process.returncode}")

finally:
    print("Shutting down ml_service...")
    process.terminate()
    process.wait()
    print("Cleanup complete.")
