#!/usr/bin/env python3
"""
Test script to check S3 data download functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Xray.pipeline.training_pipeline import TrainPipeline
from Xray.logger import logging

def test_s3_download():
    """Test if data can be downloaded from S3"""
    try:
        print("🚀 Starting S3 download test...")
        print(f"📁 Current directory: {os.getcwd()}")
        
        # Create pipeline instance
        pipeline = TrainPipeline()
        
        # Print configuration
        print(f"📦 S3 Bucket: {pipeline.data_ingestion_config.bucket_name}")
        print(f"📁 S3 Folder: {pipeline.data_ingestion_config.s3_data_folder}")
        print(f"💾 Local data path: {pipeline.data_ingestion_config.data_path}")
        
        # Test data ingestion
        print("\n⬇️  Attempting to download data from S3...")
        data_artifact = pipeline.start_data_ingestion()
        
        print("✅ S3 download completed successfully!")
        print(f"📂 Train data path: {data_artifact.train_file_path}")
        print(f"📂 Test data path: {data_artifact.test_file_path}")
        
        # Check if files were actually downloaded
        if os.path.exists(data_artifact.train_file_path):
            print(f"✅ Train directory exists: {data_artifact.train_file_path}")
            train_files = os.listdir(data_artifact.train_file_path)
            print(f"📊 Found {len(train_files)} items in train directory")
        else:
            print(f"❌ Train directory not found: {data_artifact.train_file_path}")
            
        if os.path.exists(data_artifact.test_file_path):
            print(f"✅ Test directory exists: {data_artifact.test_file_path}")
            test_files = os.listdir(data_artifact.test_file_path)
            print(f"📊 Found {len(test_files)} items in test directory")
        else:
            print(f"❌ Test directory not found: {data_artifact.test_file_path}")
            
    except Exception as e:
        print(f"❌ Error during S3 download test: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_s3_download()
