from app.services.training_service import async_train_model
from app.utils.status_tracker import training_status
import os
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_model_status(symbol):
    model_path = os.path.join("app", "ml_models", "output", f"{symbol}_model.h5")
    scaler_path = os.path.join("app", "ml_models", "output", f"{symbol}_scaler.pkl")

    # Log the current working directory
    logger.info(f"Current working directory: {os.getcwd()}")

    # Resolve absolute paths for debugging
    model_path = os.path.abspath(model_path)
    scaler_path = os.path.abspath(scaler_path)

    # Log the resolved paths and existence checks
    logger.info(f"Resolved model path: {model_path}")
    logger.info(f"Resolved scaler path: {scaler_path}")
    logger.info(f"Model exists: {os.path.exists(model_path)}")
    logger.info(f"Scaler exists: {os.path.exists(scaler_path)}")

    # Check if model and scaler exist locally
    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)

    # Dynamically verify the status
    if model_exists and scaler_exists:
        training_status.pop(symbol, None)
        return {
            "symbol": symbol,
            "model_exists": True,
            "message": "Model and scaler are available."
        }
    elif symbol in training_status and training_status[symbol] == "in_progress":
        return {
            "symbol": symbol,
            "model_exists": False,
            "message": "Training already in progress. The model will be available soon."
        }
    else:
        if not model_exists and not scaler_exists:
            async_train_model(symbol)
            training_status[symbol] = "in_progress"
            return {
                "symbol": symbol,
                "model_exists": False,
                "message": "Model for this stock code has not been trained yet. Starting training, try again later."
            }
        else:
            async_train_model(symbol)
            training_status[symbol] = "in_progress"
            return {
                "symbol": symbol,
                "model_exists": False,
                "message": "Model or scaler exists but training is not complete. Starting training and overwriting existing files."
            }