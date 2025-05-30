from app.utils.s3_utils import model_exists_in_s3

def model_exists(symbol):
    return model_exists_in_s3(symbol)