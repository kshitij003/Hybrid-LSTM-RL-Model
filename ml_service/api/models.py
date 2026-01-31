"""
Model Management API Blueprint
Handles model listing, activation, and deletion
"""

from flask import Blueprint, request, jsonify
import os
import glob
import json
from datetime import datetime
from typing import List, Dict

# Create blueprint
models_bp = Blueprint('models', __name__)

# Models directory
MODELS_DIR = "models/saved_models"


@models_bp.route('/', methods=['GET'])
@models_bp.route('/list', methods=['GET'])
def list_models():
    """
    List all available trained models
    
    Query Parameters:
    - limit: Max number of models to return (default: 10)
    - sortBy: Sort field - 'createdDate' or 'performance' (default: 'createdDate')
    
    Response:
    {
        "models": [
            {
                "modelId": "multi_stock_v1",
                "version": "1.0.0",
                "stocks": ["AAPL", "MSFT", "GOOGL"],
                "trainedOn": "2024-01-31T10:00:00Z",
                "performance": {...},
                "isActive": true
            }
        ],
        "total": 5
    }
    """
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Find all model files (.zip for PPO models)
        model_files = glob.glob(os.path.join(MODELS_DIR, "*.zip"))
        
        models_list = []
        active_model_id = get_active_model_id()
        
        for model_file in model_files:
            model_id = os.path.splitext(os.path.basename(model_file))[0]
            
            # Get model metadata
            metadata = get_model_metadata(model_id)
            
            model_info = {
                "modelId": model_id,
                "version": metadata.get("version", "1.0.0"),
                "stocks": metadata.get("stocks", []),
                "trainedOn": metadata.get("trainedOn", get_file_modified_time(model_file)),
                "filePath": model_file,
                "sizeBytes": os.path.getsize(model_file),
                "isActive": model_id == active_model_id
            }
            
            # Add performance if available
            if "performance" in metadata:
                model_info["performance"] = metadata["performance"]
            
            models_list.append(model_info)
        
        # Sort models
        sort_by = request.args.get('sortBy', 'createdDate')
        if sort_by == 'createdDate':
            models_list.sort(key=lambda x: x['trainedOn'], reverse=True)
        
        # Limit results
        limit = int(request.args.get('limit', 10))
        models_list = models_list[:limit]
        
        return jsonify({
            "models": models_list,
            "total": len(model_files)
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        }), 500


@models_bp.route('/<model_id>', methods=['GET'])
def get_model_details(model_id: str):
    """
    Get detailed information about a specific model
    
    Response:
    {
        "modelId": "multi_stock_v1",
        "version": "1.0.0",
        "stocks": ["AAPL", "MSFT", "GOOGL"],
        "trainedOn": "2024-01-31T10:00:00Z",
        "trainingConfig": {...},
        "performance": {...},
        "isActive": true
    }
    """
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
        
        if not os.path.exists(model_path):
            return jsonify({
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Model '{model_id}' not found"
                }
            }), 404
        
        # Get metadata
        metadata = get_model_metadata(model_id)
        active_model_id = get_active_model_id()
        
        model_details = {
            "modelId": model_id,
            "version": metadata.get("version", "1.0.0"),
            "stocks": metadata.get("stocks", []),
            "trainedOn": metadata.get("trainedOn", get_file_modified_time(model_path)),
            "trainingConfig": metadata.get("trainingConfig", {}),
            "performance": metadata.get("performance", {}),
            "filePath": model_path,
            "sizeBytes": os.path.getsize(model_path),
            "isActive": model_id == active_model_id
        }
        
        return jsonify(model_details), 200
    
    except Exception as e:
        return jsonify({
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        }), 500


@models_bp.route('/activate/<model_id>', methods=['POST'])
def activate_model(model_id: str):
    """
    Activate a model for inference
    
    Response:
    {
        "modelId": "multi_stock_v1",
        "status": "ACTIVATED",
        "message": "Model activated successfully",
        "previousActiveModel": "multi_stock_v0"
    }
    """
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
        
        if not os.path.exists(model_path):
            return jsonify({
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Model '{model_id}' not found"
                }
            }), 404
        
        # Get previous active model
        previous_active = get_active_model_id()
        
        # Set new active model
        set_active_model_id(model_id)
        
        # Reload inference model
        from api.inference import load_active_model
        load_active_model()
        
        return jsonify({
            "modelId": model_id,
            "status": "ACTIVATED",
            "message": "Model activated successfully",
            "previousActiveModel": previous_active
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        }), 500


@models_bp.route('/<model_id>', methods=['DELETE'])
def delete_model(model_id: str):
    """
    Delete a model
    
    Response:
    {
        "modelId": "multi_stock_v1",
        "status": "DELETED",
        "message": "Model deleted successfully"
    }
    """
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_id}.zip")
        
        if not os.path.exists(model_path):
            return jsonify({
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Model '{model_id}' not found"
                }
            }), 404
        
        # Check if it's the active model
        active_model_id = get_active_model_id()
        if model_id == active_model_id:
            return jsonify({
                "error": {
                    "code": "INVALID_OPERATION",
                    "message": "Cannot delete the active model. Activate a different model first."
                }
            }), 400
        
        # Delete model file
        os.remove(model_path)
        
        # Delete metadata file if exists
        metadata_path = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        return jsonify({
            "modelId": model_id,
            "status": "DELETED",
            "message": "Model deleted successfully"
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        }), 500


def get_model_metadata(model_id: str) -> dict:
    """Load model metadata from JSON file"""
    metadata_path = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    return {}


def save_model_metadata(model_id: str, metadata: dict):
    """Save model metadata to JSON file"""
    metadata_path = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_active_model_id() -> str:
    """Get the currently active model ID"""
    active_file = os.path.join(MODELS_DIR, "active_model.txt")
    
    if os.path.exists(active_file):
        with open(active_file, 'r') as f:
            return f.read().strip()
    
    # Default to ppo_multi_stock if exists
    default_path = os.path.join(MODELS_DIR, "ppo_multi_stock.zip")
    if os.path.exists(default_path):
        return "ppo_multi_stock"
    
    return ""


def set_active_model_id(model_id: str):
    """Set the active model ID"""
    active_file = os.path.join(MODELS_DIR, "active_model.txt")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(active_file, 'w') as f:
        f.write(model_id)


def get_file_modified_time(file_path: str) -> str:
    """Get file modification time as ISO string"""
    mtime = os.path.getmtime(file_path)
    return datetime.fromtimestamp(mtime).isoformat()
