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

# Set seed for reproducibility
seed = int(os.getenv("SEED", 42))
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

OUTPUT_DIR = os.path.join("app", "ml_models", "output")

def async_train_model(symbol):
    current_time = time.time()

    # Check if training is already in progress and if it has timed out
    if symbol in training_status:
        status, timestamp = training_status[symbol]
        model_path = os.path.join(OUTPUT_DIR, f"{symbol}_model.h5")
        scaler_path = os.path.join(OUTPUT_DIR, f"{symbol}_scaler.pkl")

        # Revalidate existence of model and scaler
        model_exists = os.path.exists(model_path)
        scaler_exists = os.path.exists(scaler_path)

        if not (model_exists and scaler_exists):
            training_status.pop(symbol, None)
        elif status == "in_progress" and (current_time - timestamp) < 3600:  # 1 hour timeout
            return
        elif status == "in_progress":
            print(f"Training for {symbol} timed out. Resetting status.")

    # Update training status
    training_status[symbol] = ("in_progress", current_time)

    def train():
        try:
            train_model(symbol)
            training_status[symbol] = ("completed", time.time())
        except Exception as e:
            print(f"Error while training {symbol}: {e}")
            training_status[symbol] = ("failed", time.time())

    thread = threading.Thread(target=train)
    thread.start()

def train_model(symbol):
    # Download data
    start_date = '1901-01-01'
    df = fetch_stock_data(symbol, period="1y")
    df.index = pd.to_datetime(df.index)

    # Get the last date from the Yahoo Finance data
    last_date = df.index.max().strftime('%Y-%m-%d')

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    # Save the scaler to a file
    scaler_path = os.path.join(OUTPUT_DIR, f"{symbol}_scaler.pkl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Create sequences
    def create_sequences(data, window_size=60):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i, :])
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
        verbose=1
    )
    training_duration = time.time() - start_time

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

    # Save metrics to a JSON file
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R²": r2,
        "last_data_date": last_date,
        "training_duration": training_duration,
        "training_data_size": len(X_train),
        "validation_data_size": len(X_val)
    }
    metrics_path = os.path.join(OUTPUT_DIR, f"{symbol}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    # Save model to a file
    model_path = os.path.join(OUTPUT_DIR, f"{symbol}_model.h5")
    model.save(model_path)

def clean_training_status():
    invalid_symbols = []
    for symbol, (status, _) in training_status.items():
        model_path = os.path.join(OUTPUT_DIR, f"{symbol}_model.h5")
        scaler_path = os.path.join(OUTPUT_DIR, f"{symbol}_scaler.pkl")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            invalid_symbols.append(symbol)

    for symbol in invalid_symbols:
        training_status.pop(symbol, None)
