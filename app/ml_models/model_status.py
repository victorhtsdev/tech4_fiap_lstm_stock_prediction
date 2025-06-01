from app.services.training_service import async_train_model
from app.services.amazon.s3_service import S3Manager
from app.utils.status_tracker import training_status
import os

def handle_model_status(symbol):
    model_key = f"models/{symbol}/model.h5"
    scaler_key = f"models/{symbol}/scaler.pkl"

    # Check if model and scaler exist in S3
    s3_manager = S3Manager(bucket_name=os.getenv("AWS_S3_BUCKET"), region_name=os.getenv("AWS_REGION"))
    model_exists = s3_manager.check_model(model_key)
    scaler_exists = s3_manager.check_model(scaler_key)

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