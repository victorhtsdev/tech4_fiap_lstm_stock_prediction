from flask import Blueprint, request, jsonify
from app.services.training_service import async_train_model
from app.utils.logger import AppLogger
from app.utils.status_tracker import training_status
from app.services.predicition_service import ModelPredictor
from app.ml_models.model_status import handle_model_status
import os
import json

# Initialize logger
logger = AppLogger(__name__).get_logger()

bp = Blueprint('predict_routes', __name__)

# Initialize ModelPredictor with the directory where models are stored
model_predictor = ModelPredictor(
    bucket_name="lstm-models-bucket",
    region_name="us-east-1",
    model_dir=os.getenv("MODEL_DIR", "./app/ml_models/output")
)

@bp.route('/models/check', methods=['POST'])
def check_models():
    data = request.get_json()
    symbols = data.get("symbols", [])

    results = []

    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        results.append(handle_model_status(symbol))

    return jsonify(results), 200

@bp.route('/models/metrics', methods=['POST'])
def get_models_metrics():
    data = request.get_json()
    symbols = data.get("symbols", [])

    results = []

    for symbol in symbols:
        symbol = symbol.upper()
        metrics_path = os.path.join("./app/ml_models/output", f"{symbol}_metrics.json")
        if not os.path.exists(metrics_path):
            results.append({
                "symbol": symbol,
                "error": "Metrics for this model are not available."
            })
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        logger.info(f"Successfully retrieved metrics for {symbol}: {metrics}")

        results.append({
            "symbol": symbol,
            "metrics": {
                "MAE": metrics.get("MAE"),
                "RMSE": metrics.get("RMSE"),
                "MAPE": metrics.get("MAPE"),
                "R²": metrics.get("R²"),
                "last_data_date": metrics.get("last_data_date"),
                "training_duration": metrics.get("training_duration"),
                "training_data_size": metrics.get("training_data_size"),
                "validation_data_size": metrics.get("validation_data_size")
            }
        })

    return jsonify(results), 200

@bp.route('/models/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbols = data.get("symbols")

    if not symbols or not isinstance(symbols, list):
        return jsonify({"error": "Symbols parameter must be a list of stock symbols."}), 400

    results = {}

    for symbol in symbols:
        status = handle_model_status(symbol)
        if not status["model_exists"]:
            results[symbol] = {
                "error": status["message"],
                "status": "failed"
            }
        else:
            try:
                predictions = model_predictor.predict([symbol])
                results[symbol] = predictions[symbol]
            except Exception as e:
                logger.error(f"Error during prediction for symbol {symbol}: {e}")
                results[symbol] = {
                    "error": "An error occurred during prediction.",
                    "status": "failed"
                }

    return jsonify({"predictions": results}), 200