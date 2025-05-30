from flask import Blueprint, request, jsonify
from app.services.training_service import async_train_model
from app.services.amazon.s3_service import S3Manager
from app.utils.logger import AppLogger
from app.utils.status_tracker import training_status
import os
import json

# Initialize logger
logger = AppLogger(__name__).get_logger()

bp = Blueprint('predict_routes', __name__)

@bp.route('/models/check', methods=['POST'])
def check_models():
    data = request.get_json()
    symbols = data.get("symbols", [])

    results = []

    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        model_key = f"models/{symbol}/model.h5"
        scaler_key = f"models/{symbol}/scaler.pkl"

        # Check if model and scaler exist in S3
        s3_manager = S3Manager(bucket_name=os.getenv("AWS_S3_BUCKET"), region_name=os.getenv("AWS_REGION"))
        model_exists = s3_manager.check_model(model_key)
        scaler_exists = s3_manager.check_model(scaler_key)

        # Dynamically verify the status
        if model_exists and scaler_exists:
            results.append({
                "symbol": symbol,
                "model_exists": True,
                "message": "Model and scaler are available."
            })
            # Clean up training status if model exists
            training_status.pop(symbol, None)
        elif symbol in training_status and training_status[symbol] == "in_progress":
            results.append({
                "symbol": symbol,
                "model_exists": False,
                "message": "Training already in progress. The model will be available soon."
            })
        else:
            # Check again to ensure model and scaler are missing before starting training
            if not model_exists and not scaler_exists:
                results.append({
                    "symbol": symbol,
                    "model_exists": False,
                    "message": "Model for this stock code has not been trained yet. Starting training, try again later."
                })
                async_train_model(symbol)
                training_status[symbol] = "in_progress"
            else:
                # If model or scaler exists but training is not complete, start training and overwrite
                results.append({
                    "symbol": symbol,
                    "model_exists": False,
                    "message": "Model or scaler exists but training is not complete. Starting training and overwriting existing files."
                })
                async_train_model(symbol)
                training_status[symbol] = "in_progress"

    return jsonify(results), 200

@bp.route('/models/metrics', methods=['POST'])
def get_models_metrics():
    data = request.get_json()
    symbols = data.get("symbols", [])

    results = []

    for symbol in symbols:
        symbol = symbol.upper()
        s3_key = f"models/{symbol}/metrics.json"

        try:
            # Use S3Manager to check and download the metrics file
            s3_manager = S3Manager(bucket_name=os.getenv("AWS_S3_BUCKET"), region_name=os.getenv("AWS_REGION"))
            if not s3_manager.check_model(s3_key):
                results.append({
                    "symbol": symbol,
                    "error": "Metrics for this model are not available."
                })
                continue

            # Download the metrics file directly into memory
            metrics_object = s3_manager.s3_client.get_object(Bucket=os.getenv("AWS_S3_BUCKET"), Key=s3_key)
            metrics = json.loads(metrics_object['Body'].read().decode('utf-8'))

            logger.info(f"Successfully retrieved metrics for {symbol}: {metrics}")

            results.append({
                "symbol": symbol,
                "metrics": metrics
            })

        except Exception as e:
            logger.error(f"Error retrieving metrics for {symbol}: {e}")
            results.append({
                "symbol": symbol,
                "error": "An error occurred while retrieving the metrics."
            })

    return jsonify(results), 200