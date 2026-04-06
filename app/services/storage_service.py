import os
import logging

import boto3
from botocore.exceptions import ClientError
from config import Config

logger = logging.getLogger(__name__)


class LocalStorage:
    """Store uploaded files on the local filesystem."""

    def __init__(self, base_dir: str = None):
        self.base_dir = os.path.abspath(base_dir or Config.LOCAL_UPLOAD_DIR)
        os.makedirs(self.base_dir, exist_ok=True)

    def upload_file(self, file_obj, filename: str) -> bool:
        try:
            safe_name = os.path.basename(filename)
            dest = os.path.join(self.base_dir, safe_name)
            with open(dest, "wb") as f:
                f.write(file_obj.read())
            logger.info(f"File saved locally: {dest}")
            return True
        except Exception as e:
            logger.error(f"Error saving file locally: {e}")
            return False

    def get_file(self, filename: str):
        try:
            safe_name = os.path.basename(filename)
            path = os.path.join(self.base_dir, safe_name)
            return open(path, "rb")
        except Exception as e:
            logger.error(f"Error reading local file: {e}")
            return None

    def list_files(self) -> list:
        try:
            return os.listdir(self.base_dir)
        except Exception as e:
            logger.error(f"Error listing local files: {e}")
            return []


class S3Storage:
    """Store uploaded files in an AWS S3 bucket."""

    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_ACCESS_KEY,
            aws_secret_access_key=Config.AWS_SECRET_KEY,
            region_name=Config.AWS_REGION,
        )
        self.bucket = Config.AWS_BUCKET_NAME

    def upload_file(self, file_obj, filename: str) -> bool:
        try:
            self.s3.upload_fileobj(file_obj, self.bucket, filename)
            return True
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            return False

    def get_file(self, filename: str):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=filename)
            return response['Body']
        except ClientError as e:
            logger.error(f"Error retrieving file from S3: {e}")
            return None

    def list_files(self, prefix: str = "") -> list:
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError as e:
            logger.error(f"Error listing S3 files: {e}")
            return []


def get_storage_service(backend: str = None, local_dir: str = None):
    """Factory: return the appropriate storage service based on configuration.

    Args:
        backend: "s3" or "local". Falls back to Config.STORAGE_BACKEND.
        local_dir: Override local directory (only used when backend="local").
    """
    backend = (backend or Config.STORAGE_BACKEND).lower()
    if backend == "s3":
        return S3Storage()
    return LocalStorage(base_dir=local_dir)

