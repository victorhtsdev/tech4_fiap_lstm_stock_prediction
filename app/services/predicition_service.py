from app.utils.s3_utils import model_exists_in_s3
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
import os
import pickle
from app.services.amazon.s3_service import S3Manager
from io import BytesIO
import tempfile
from app.services.data_service import fetch_stock_data, get_cached_stock_data
import json
from datetime import datetime
import threading

def model_exists(symbol):
    return model_exists_in_s3(symbol)

class ModelPredictor:
    def __init__(self, bucket_name=None, region_name=None, model_dir=None):
        if bucket_name and region_name:
            self.s3_manager = S3Manager(bucket_name=bucket_name, region_name=region_name)
        self.model_dir = model_dir

    def load_model_and_scaler(self, symbol):
        model_key = f"models/{symbol}/model.h5"
        scaler_key = f"models/{symbol}/scaler.pkl"

        # Define paths within the ml_models temp directory
        temp_model_path = os.path.join(TEMP_DIR, f"{symbol}_model.h5")
        temp_scaler_path = os.path.join(TEMP_DIR, f"{symbol}_scaler.pkl")

        # Check if model exists locally, if not download from S3
        if not os.path.exists(temp_model_path):
            model_object = self.s3_manager.s3_client.get_object(Bucket=self.s3_manager.bucket_name, Key=model_key)
            with open(temp_model_path, 'wb') as temp_model_file:
                temp_model_file.write(model_object['Body'].read())

        # Load model
        model = load_model(temp_model_path)

        # Check if scaler exists locally, if not download from S3
        if not os.path.exists(temp_scaler_path):
            scaler_object = self.s3_manager.s3_client.get_object(Bucket=self.s3_manager.bucket_name, Key=scaler_key)
            with open(temp_scaler_path, 'wb') as temp_scaler_file:
                temp_scaler_file.write(scaler_object['Body'].read())

        # Load scaler
        with open(temp_scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        return model, scaler

    def get_last_60_days_data(self, symbol):
        try:
            # Try to get cached data first
            data = get_cached_stock_data(symbol)
        except ValueError:
            # If cache is not available or outdated, fetch fresh data
            data = fetch_stock_data(symbol, period="1y")

            # Save the fetched data to cache for future use
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models", "temp")
            cache_file = os.path.join(temp_dir, f"{symbol}_data.json")
            data.reset_index(inplace=True)  # Ensure the index is included as a column
            data.to_json(cache_file, orient="records", date_format="iso")
            data.set_index("Date", inplace=True)  # Restore the index as Date

        return data[-60:]

    def get_prediction_cache(self, symbol):
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models", "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        cache_file = os.path.join(temp_dir, f"{symbol}_prediction.json")
        today = datetime.now().strftime("%Y-%m-%d")

        # Check if cached prediction exists and is up-to-date
        if os.path.exists(cache_file):
            modified_date = datetime.fromtimestamp(os.path.getmtime(cache_file)).strftime("%Y-%m-%d")
            if modified_date == today:
                with open(cache_file, "r") as f:
                    return json.load(f)

        return None

    def save_prediction_cache(self, symbol, prediction):
        def save_to_file():
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models", "temp")
            cache_file = os.path.join(temp_dir, f"{symbol}_prediction.json")

            with open(cache_file, "w") as f:
                json.dump(prediction, f)

        # Run the save operation in a separate thread
        save_thread = threading.Thread(target=save_to_file)
        save_thread.start()

    def predict(self, symbols):
        predictions = {}

        for symbol in symbols:
            try:
                # Check prediction cache first
                cached_prediction = self.get_prediction_cache(symbol)
                if cached_prediction:
                    predictions[symbol] = cached_prediction
                    continue

                model, scaler = self.load_model_and_scaler(symbol)
                data = self.get_last_60_days_data(symbol)

                # Prepare data for prediction
                scaled_data = scaler.transform(data.values)
                input_data = np.array([scaled_data[-60:]])

                # Make prediction
                prediction = model.predict(input_data)
                # Reverse scaling for the 'Close' column
                close_min = scaler.data_min_[0]
                close_max = scaler.data_max_[0]
                predicted_price = prediction[0][0] * (close_max - close_min) + close_min

                # Calculate variation from the last close price
                last_close_price = float(data['Close'].iloc[-1])
                variation = predicted_price - last_close_price
                variation_percentage = (variation / last_close_price) * 100
                variation_status = "positive" if variation > 0 else "negative"

                prediction_result = {
                    "predicted_price": predicted_price,
                    "status": "success",
                    "variation": {
                        "status": variation_status,
                        "percentage": variation_percentage
                    }
                }

                # Save prediction to cache
                self.save_prediction_cache(symbol, prediction_result)

                predictions[symbol] = prediction_result
            except Exception as e:
                predictions[symbol] = {
                    "error": str(e),
                    "status": "failed"
                }

        return predictions

# Define a temporary directory within the ml_models directory
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models", "temp")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)