from flask import Blueprint, request, jsonify
from app.services.training_service import async_train_model
from app.utils.logger import AppLogger
from app.utils.status_tracker import training_status
from app.services.predicition_service import ModelPredictor
from app.ml_models.model_status import handle_model_status
import os
import json
from flask_swagger_ui import get_swaggerui_blueprint

# Initialize logger
logger = AppLogger(__name__).get_logger()

bp = Blueprint('predict_routes', __name__)

# Initialize ModelPredictor with the directory where models are stored
model_predictor = ModelPredictor(
    bucket_name="lstm-models-bucket",
    region_name="us-east-1",
    model_dir=os.getenv("MODEL_DIR", "./app/ml_models/output")
)

SWAGGER_URL = '/swagger'
API_URL = '/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "LSTM Stock Prediction API"
    }
)

bp.add_url_rule(SWAGGER_URL, 'swagger_ui', view_func=swaggerui_blueprint)

@bp.route('/swagger.json', methods=['GET'])
def swagger_spec():
    """Serve the Swagger specification."""
    swagger_spec = {
        "swagger": "2.0",
        "info": {
            "version": "1.0.0",
            "title": "LSTM Stock Prediction API",
            "description": "API for stock price prediction and model management."
        },
        "host": "localhost:5000",
        "basePath": "/",
        "schemes": ["http"],
        "paths": {
            "/models/check": {
                "post": {
                    "summary": "Check model status",
                    "description": "Check the status of models for the given stock symbols. If a model does not exist for a symbol, training will be initiated automatically.",
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "symbols": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["symbols"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "schema": {
                                "type": "array",
                                "items": {"type": "object"}
                            }
                        }
                    }
                }
            },
            "/models/metrics": {
                "post": {
                    "summary": "Get model metrics",
                    "description": "Retrieve metrics for the given stock symbols.",
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "symbols": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["symbols"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "schema": {
                                "type": "array",
                                "items": {"type": "object"}
                            }
                        }
                    }
                }
            },
            "/models/predict": {
                "post": {
                    "summary": "Predict stock prices",
                    "description": "Predict stock prices for the given symbols. If a model does not exist for a symbol, training will be initiated automatically.",
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "symbols": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["symbols"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "predictions": {
                                        "type": "object"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/models/retrain": {
                "post": {
                    "summary": "Retrain a model",
                    "description": "Force retraining of a model for the given symbol.",
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"}
                                },
                                "required": ["symbol"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return jsonify(swagger_spec)

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

@bp.route('/models/retrain', methods=['POST'])
def retrain_model():
    data = request.get_json()
    symbol = data.get("symbol")

    if not symbol:
        return jsonify({"error": "Symbol parameter is required."}), 400

    try:
        # Chama o serviço de treinamento para re-treinar o modelo
        async_train_model(symbol.upper())
        logger.info(f"Re-training initiated for model: {symbol.upper()}")
        return jsonify({"message": f"Re-training initiated for model: {symbol.upper()}"}), 200
    except Exception as e:
        logger.error(f"Error initiating re-training for model {symbol.upper()}: {e}")
        return jsonify({"error": "An error occurred while initiating re-training."}), 500