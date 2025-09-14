#!/usr/bin/env python3
"""
Test script for Model Pusher functionality.

This script demonstrates how to use the ModelPusher class to:
1. Package a trained model with BentoML
2. Create a prediction service
3. Prepare deployment artifacts

Usage:
    python test_model_pusher.py
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from Xray.pipeline.training_pipeline import TrainPipeline
from Xray.logger import logging

def test_model_pusher():
    """
    Test the complete model pusher functionality.
    """
    logging.info("=" * 50)
    logging.info("TESTING MODEL PUSHER FUNCTIONALITY")
    logging.info("=" * 50)

    try:
        # Initialize the training pipeline
        train_pipeline = TrainPipeline()

        # Run the complete pipeline including model pusher
        logging.info("Running complete pipeline with model pusher...")
        train_pipeline.run_pipeline()

        logging.info("✅ Model pusher test completed successfully!")
        logging.info("Check the 'artifacts/model_deployment' directory for deployment artifacts")

    except Exception as e:
        logging.error(f"❌ Model pusher test failed: {str(e)}")
        raise e

def test_model_pusher_only():
    """
    Test only the model pusher step (requires existing trained model).
    """
    logging.info("=" * 50)
    logging.info("TESTING MODEL PUSHER ONLY")
    logging.info("=" * 50)

    try:
        # This would require existing artifacts from previous runs
        # For now, we'll just show the structure
        logging.info("To test model pusher only, you need:")
        logging.info("1. A trained model in artifacts/[timestamp]/model_training/model.pt")
        logging.info("2. Data transformation artifacts")
        logging.info("3. Run the complete pipeline first")

    except Exception as e:
        logging.error(f"❌ Model pusher test failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test the complete pipeline with model pusher
    test_model_pusher()

    # Uncomment to test only model pusher (requires existing artifacts)
    # test_model_pusher_only()
