import os
import threading
import time
from datetime import datetime, timedelta
from app.services.predicition_service import ModelPredictor
from app.services.training_service import async_train_model
from app.utils.logger import AppLogger

logger = AppLogger(__name__).get_logger()

# Initialize the ModelPredictor instance
predictor = ModelPredictor()

def list_symbols(output_folder):
    """List all action codes based on model files in the output folder."""
    symbols = []
    for file in os.listdir(output_folder):
        if file.endswith("_model.h5"):
            symbol = file.split("_model.h5")[0]
            symbols.append(symbol)
    return symbols

def check_and_train_model(symbol, model_path):
    """Check if the model needs retraining and trigger training if necessary."""
    last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
    if datetime.now() - last_modified > timedelta(days=15):
        logger.info(f"Model for {symbol} is older than 15 days. Triggering training.")
        async_train_model(symbol)

def run_daily_tasks():
    """Run daily tasks: predict and check model retraining."""
    output_folder = os.path.join(os.getcwd(), "app", "ml_models", "output")
    symbols = list_symbols(output_folder)

    for symbol in symbols:
        try:
            # Run prediction
            logger.info(f"Running prediction for {symbol}.")
            predictor.predict([symbol])

            # Check and train model if necessary
            model_path = os.path.join(output_folder, f"{symbol}_model.h5")
            check_and_train_model(symbol, model_path)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

def start_daily_tasks_thread():
    """Start the daily tasks in a separate thread."""
    def task_runner():
        while True:
            try:
                logger.info("Starting daily tasks.")
                run_daily_tasks()
                logger.info("Daily tasks completed. Sleeping until the next scheduled run.")

                # Calculate sleep time until the next scheduled hour
                now = datetime.now()
                next_run = now.replace(hour=int(os.getenv("DAILY_TASK_HOUR", 0)), minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                sleep_time = (next_run - now).total_seconds()
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error in daily tasks thread: {e}")

    # Run immediately on server start
    threading.Thread(target=run_daily_tasks, daemon=True).start()

    # Start the scheduled task runner
    thread = threading.Thread(target=task_runner, daemon=True)
    thread.start()