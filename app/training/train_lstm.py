import yfinance as yf
import pandas as pd
import numpy as np
import boto3
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm_model(symbol='PETR4.SA', bucket=None, region='us-east-1'):
    model_file = 'model.h5'
    s3_key = f"{symbol}/{model_file}"

    df = yf.download(symbol, start='2000-01-01')
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    window_size = 60
    X, y = [], []

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=10, batch_size=32, verbose=1,
              callbacks=[EarlyStopping(monitor='loss', patience=3)])

    model.save(model_file)

    s3 = boto3.client('s3', region_name=region)
    s3.upload_file(model_file, bucket, s3_key)
    print(f"Model uploaded to s3://{bucket}/{s3_key}")
