#!/usr/bin/env python3
"""
Test script to verify model evaluation component works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Xray.pipeline.training_pipeline import TrainPipeline
from Xray.logger import logging

def test_model_evaluation():
    """Test the complete pipeline including model evaluation"""
    try:
        logging.info("=" * 50)
        logging.info("STARTING MODEL EVALUATION TEST")
        logging.info("=" * 50)

        # Create pipeline instance
        pipeline = TrainPipeline()

        # Run the complete pipeline (data ingestion + transformation + training + evaluation)
        logging.info("Running complete pipeline with evaluation...")
        pipeline.run_pipeline()

        logging.info("=" * 50)
        logging.info("MODEL EVALUATION TEST COMPLETED SUCCESSFULLY!")
        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"Model evaluation test failed: {str(e)}")
        raise e

if __name__ == "__main__":
    test_model_evaluation()
