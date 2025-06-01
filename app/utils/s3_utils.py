import os

def model_exists_locally(symbol):
    """Check if the model and scaler exist locally in the output directory."""
    output_dir = os.path.join("ml_models", "output")
    model_path = os.path.join(output_dir, f"{symbol}_model.h5")
    scaler_path = os.path.join(output_dir, f"{symbol}_scaler.pkl")

    return os.path.exists(model_path) and os.path.exists(scaler_path)
