import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime

def fetch_stock_data(symbol, period="1y", columns=["Close", "High", "Low", "Open"]):
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbol (str): Stock symbol to fetch data for.
        period (str): Period of data to fetch (default is 1 year).
        columns (list): List of columns to keep (default is ["Close", "High", "Low", "Open"]).

    Returns:
        pd.DataFrame: DataFrame containing the requested stock data.

    Raises:
        ValueError: If the fetched data is insufficient or empty.
    """
    data = yf.download(symbol, period=period)
    if data.empty:
        raise ValueError(f"No data found for symbol {symbol}.")

    data = data[columns].dropna()
    if len(data) < 60:
        raise ValueError(f"Not enough data to make a prediction for {symbol}. At least 60 days of data are required.")

    return data

def get_cached_stock_data(symbol):
    """
    Retrieve stock data from the local cache if available and up-to-date.

    Args:
        symbol (str): Stock symbol to retrieve data for.

    Returns:
        pd.DataFrame: DataFrame containing the cached stock data.

    Raises:
        ValueError: If the cache is outdated or missing.
    """
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models", "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    cache_file = os.path.join(temp_dir, f"{symbol}_data.json")
    today = datetime.now().strftime("%Y-%m-%d")

    # Check if cached data exists and is up-to-date
    if os.path.exists(cache_file):
        modified_date = datetime.fromtimestamp(os.path.getmtime(cache_file)).strftime("%Y-%m-%d")
        if modified_date == today:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            data = pd.DataFrame(cached_data)

            # Ensure the 'Date' column is present and set as the index
            if "Date" in data.columns:
                data.set_index("Date", inplace=True)
            else:
                raise ValueError(f"Cached data for {symbol} is missing the 'Date' column.")

            return data

    raise ValueError(f"Cached data for {symbol} is not available or outdated.")

def get_prediction_cache(symbol, temp_dir):
    """
    Retrieve prediction from the local cache if available.

    Args:
        symbol (str): Stock symbol to retrieve prediction for.
        temp_dir (str): Path to the temporary directory.

    Returns:
        dict: Cached prediction result if available, otherwise None.
    """
    cache_file = os.path.join(temp_dir, f"{symbol}_prediction.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    return None

def save_prediction_cache(symbol, prediction_result, temp_dir):
    """
    Save prediction result to the local cache.

    Args:
        symbol (str): Stock symbol to save prediction for.
        prediction_result (dict): Prediction result to save.
        temp_dir (str): Path to the temporary directory.
    """
    cache_file = os.path.join(temp_dir, f"{symbol}_prediction.json")

    with open(cache_file, "w") as f:
        json.dump(prediction_result, f)

def save_stock_data(symbol, data, temp_dir):
    """
    Save stock data to the local cache.

    Args:
        symbol (str): Stock symbol to save data for.
        data (pd.DataFrame): Stock data to save.
        temp_dir (str): Path to the temporary directory.
    """
    cache_file = os.path.join(temp_dir, f"{symbol}_data.json")

    # Ensure the 'Date' column is included as a regular column
    data.reset_index(inplace=True)
    data.to_json(cache_file, orient="records", date_format="iso")
    data.set_index("Date", inplace=True)  # Restore the index as Date

def get_stock_data(symbol, temp_dir):
    """
    Retrieve stock data from the local cache if available and up-to-date.

    Args:
        symbol (str): Stock symbol to retrieve data for.
        temp_dir (str): Path to the temporary directory.

    Returns:
        pd.DataFrame: DataFrame containing the cached stock data.

    Raises:
        ValueError: If the cache is outdated or missing.
    """
    cache_file = os.path.join(temp_dir, f"{symbol}_data.json")
    today = datetime.now().strftime("%Y-%m-%d")

    # Check if cached data exists and is up-to-date
    if os.path.exists(cache_file):
        modified_date = datetime.fromtimestamp(os.path.getmtime(cache_file)).strftime("%Y-%m-%d")
        if modified_date == today:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            data = pd.DataFrame(cached_data)

            # Ensure the 'Date' column is present and set as the index
            if "Date" in data.columns:
                data.set_index("Date", inplace=True)
            else:
                raise ValueError(f"Cached data for {symbol} is missing the 'Date' column.")

            return data

    raise ValueError(f"Cached data for {symbol} is not available or outdated.")
