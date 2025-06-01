from app.utils.logger import AppLogger
from app.services.amazon.s3_service import S3Manager
import pickle
import random
import tensorflow as tf
import json
import time
from app.utils.status_tracker import training_status
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.ml_models.model_factory import LSTMModelFactory
import os
import numpy as np
import threading
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from app.services.data_service import fetch_stock_data

logger = AppLogger(__name__).get_logger()

# Configurações do S3
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_REGION = os.getenv("AWS_REGION")

# Configuração do S3Manager
s3_manager = S3Manager(bucket_name=S3_BUCKET, region_name=S3_REGION)

# Set seed for reproducibility
seed = int(os.getenv("SEED", 42))
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

def async_train_model(symbol):
    current_time = time.time()

    logger.info(f"Checking training status for {symbol}...")

    # Check if training is already in progress and if it has timed out
    if symbol in training_status:
        status, timestamp = training_status[symbol]
        logger.info(f"Current status for {symbol}: {status}")
        model_key = f"models/{symbol}/model.h5"
        scaler_key = f"models/{symbol}/scaler.pkl"

        # Revalidate existence of model and scaler in S3
        model_exists = s3_manager.check_model(model_key)
        scaler_exists = s3_manager.check_model(scaler_key)

        if not (model_exists and scaler_exists):
            logger.warning(f"Model or scaler for {symbol} no longer exists in S3. Resetting status.")
            training_status.pop(symbol, None)
        elif status == "in_progress" and (current_time - timestamp) < 3600:  # 1 hour timeout
            logger.info(f"Training already in progress for {symbol}")
            return
        elif status == "in_progress":
            logger.warning(f"Training for {symbol} timed out. Resetting status.")

    # Update training status
    logger.info(f"Starting training for {symbol}...")
    training_status[symbol] = ("in_progress", current_time)

    def train():
        try:
            logger.info(f"Thread started for training {symbol}.")
            train_model(symbol)
            training_status[symbol] = ("completed", time.time())
            logger.info(f"Training completed for {symbol}.")
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            training_status[symbol] = ("failed", time.time())

    thread = threading.Thread(target=train)
    thread.start()

def train_model(symbol):
    logger.info(f"Starting training for {symbol}")

    # Download data
    start_date = '1901-01-01'
    df = fetch_stock_data(symbol, period="max")
    df.index = pd.to_datetime(df.index)

    # Get the last date from the Yahoo Finance data
    last_date = df.index.max().strftime('%Y-%m-%d')

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    # Save the scaler to a file
    scaler_path = f"{symbol}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Upload the scaler to S3 in the correct folder
    s3_manager.upload_model(scaler_path, f"models/{symbol}/scaler.pkl")
    os.remove(scaler_path)

    # Create sequences
    def create_sequences(data, window_size=60):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, :])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    window_size = 60
    X, y = create_sequences(scaled_data, window_size)

    # Split data into training and validation sets
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create model using factory
    factory = LSTMModelFactory()
    model = factory.create_model(input_shape=(window_size, 4))

    # Train model
    start_time = time.time()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1  # Ensure verbose output
    )

    # Calculate training duration
    training_duration = time.time() - start_time

    # Log metrics
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    # Calculate additional metrics
    y_pred_scaled = model.predict(X_val)
    y_pred = y_pred_scaled.reshape(-1, 1)

    close_min = scaler.data_min_[0]
    close_max = scaler.data_max_[0]

    y_pred_real = y_pred * (close_max - close_min) + close_min
    y_val_real = y_val.reshape(-1, 1) * (close_max - close_min) + close_min

    mae = mean_absolute_error(y_val_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
    mape = np.mean(np.abs((y_val_real - y_pred_real) / y_val_real)) * 100
    r2 = r2_score(y_val_real, y_pred_real)

    # Ensure training and validation sizes are calculated correctly
    training_data_size = len(X_train)
    validation_data_size = len(X_val)

    # Save metrics to a JSON file
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R²": r2,
        "last_data_date": last_date,
        "training_duration": training_duration,
        "training_data_size": training_data_size,
        "validation_data_size": validation_data_size
    }
    metrics_path = f"{symbol}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    # Upload metrics to S3 in the correct folder
    s3_manager.upload_model(metrics_path, f"models/{symbol}/metrics.json")
    os.remove(metrics_path)

    # Save model to a file
    model_path = f"{symbol}_model.h5"
    model.save(model_path)

    # Upload the model to S3 in the correct folder
    s3_manager.upload_model(model_path, f"models/{symbol}/model.h5")
    os.remove(model_path)

    logger.info(f"Training completed for {symbol}")

    # Remove temporary files after uploading to S3
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(scaler_path):
        os.remove(scaler_path)

def clean_training_status():
    logger.info("Cleaning training status for invalid symbols...")
    invalid_symbols = []

    for symbol, (status, _) in training_status.items():
        model_key = f"models/{symbol}/model.h5"
        scaler_key = f"models/{symbol}/scaler.pkl"

        # Check if model and scaler exist in S3
        model_exists = s3_manager.check_model(model_key)
        scaler_exists = s3_manager.check_model(scaler_key)

        if not (model_exists and scaler_exists):
            invalid_symbols.append(symbol)

    # Remove invalid symbols from training_status
    for symbol in invalid_symbols:
        logger.info(f"Removing {symbol} from training status.")
        training_status.pop(symbol, None)


