"""
Flask ML Service for LSTM+RL Trading System
Main application entry point
"""

from flask import Flask, jsonify
from flask_cors import CORS
import os

# Create Flask app
app = Flask(__name__)

# Enable CORS for Spring Boot backend
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080"],
        "methods": ["GET", "POST", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request


# ============================================
# Health Check Endpoint
# ============================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ml-backend",
        "version": "1.0.0",
        "framework": "Flask"
    }), 200


# ============================================
# API Routes - Registering Blueprints
# ============================================
# Import blueprints
from api.inference import inference_bp, load_active_model
from api.training import training_bp
from api.news import news_bp
from api.models import models_bp

# Register blueprints
app.register_blueprint(inference_bp, url_prefix='/api')
app.register_blueprint(training_bp, url_prefix='/api/train')
app.register_blueprint(news_bp, url_prefix='/api/news')
app.register_blueprint(models_bp, url_prefix='/api/models')


# ============================================
# Error Handlers
# ============================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": {
            "code": "NOT_FOUND",
            "message": "Endpoint not found",
            "path": str(error)
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "details": str(error)
        }
    }), 500


# ============================================
# Main Entry Point
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("=" * 50)
    print(f"🚀 Starting ML Service on port {port}")
    print(f"📍 Health check: http://localhost:{port}/health")
    print(f"📍 API Docs: http://localhost:{port}/api")
    print("=" * 50)
    
    # Load active PPO model on startup
    print("\n🔄 Loading active model...")
    if load_active_model():
        print("✅ Model loaded successfully")
    else:
        print("⚠️  No trained model found - train a model first")
    
    print("\n" + "=" * 50)
    print("🎯 Available Endpoints:")
    print("  POST /api/predict - Get portfolio recommendations")
    print("  POST /api/train/multi-stock - Start training")
    print("  GET  /api/train/status/<id> - Training status")
    print("  GET  /api/models - List models")
    print("  POST /api/models/activate/<id> - Activate model")
    print("=" * 50 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
