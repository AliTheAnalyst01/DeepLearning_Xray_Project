#!/usr/bin/env python3
"""
Test script for improved model training with better accuracy
"""

import sys
import os
sys.path.append('.')

from Xray.pipeline.training_pipeline import TrainPipeline
from Xray.logger import logging

def test_improved_model():
    """
    Test the improved model training pipeline
    """
    try:
        logging.info("=" * 50)
        logging.info("STARTING IMPROVED MODEL TRAINING TEST")
        logging.info("=" * 50)

        # Initialize the training pipeline
        pipeline = TrainPipeline()

        # Run the complete pipeline
        logging.info("Running improved pipeline...")
        pipeline.run_pipeline()

        logging.info("=" * 50)
        logging.info("IMPROVED MODEL TRAINING TEST COMPLETED SUCCESSFULLY!")
        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"Improved model training test failed: {str(e)}")
        raise e

if __name__ == "__main__":
    test_improved_model()
