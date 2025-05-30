import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

def model_exists_in_s3(symbol):
    key = f"{symbol}/model.h5"
    try:
        s3.head_object(Bucket=AWS_S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        raise
