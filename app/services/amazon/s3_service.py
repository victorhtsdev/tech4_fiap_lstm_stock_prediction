import boto3
from botocore.exceptions import NoCredentialsError
from app.utils.logger import AppLogger

logger = AppLogger(__name__).get_logger()

class S3Manager:
    def __init__(self, bucket_name, region_name):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region_name)

    def upload_model(self, local_path, s3_key):
        try:
            logger.info(f"Uploading {local_path} to S3 bucket {self.bucket_name} as {s3_key}")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Upload successful: {s3_key}")
        except FileNotFoundError:
            logger.error(f"File not found: {local_path}")
            raise
        except NoCredentialsError:
            logger.error("AWS credentials not available.")
            raise
        except Exception as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise

    def check_model(self, s3_key):
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Model found in S3: {s3_key}")
            return True
        except self.s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.info(f"Model not found in S3: {s3_key}")
                return False
            else:
                logger.error(f"Error checking model in S3: {e}")
                raise