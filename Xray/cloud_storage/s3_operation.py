import boto3
import os
from botocore.exceptions import ClientError
import logging

class S3Operation:
    def __init__(self):
        self.s3_client = boto3.client('s3')
    
    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        """
        Download entire folder from S3 bucket
        
        Args:
            folder: Local directory to download to
            bucket_name: S3 bucket name
            bucket_folder_name: S3 folder path
        """
        try:
            # Create local directory if it doesn't exist
            os.makedirs(folder, exist_ok=True)
            
            # List objects in the S3 folder
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=bucket_folder_name
            )
            
            if 'Contents' not in response:
                logging.warning(f"No objects found in s3://{bucket_name}/{bucket_folder_name}")
                return
            
            # Download each file
            for obj in response['Contents']:
                s3_key = obj['Key']
                if s3_key.endswith('/'):  # Skip directories
                    continue
                
                # Create local file path
                local_file_path = os.path.join(folder, os.path.relpath(s3_key, bucket_folder_name))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download file
                self.s3_client.download_file(bucket_name, s3_key, local_file_path)
                logging.info(f"Downloaded {s3_key} to {local_file_path}")
                
        except ClientError as e:
            logging.error(f"Error syncing folder from S3: {e}")
            raise