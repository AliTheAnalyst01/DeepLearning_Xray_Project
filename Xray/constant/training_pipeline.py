import os
import time

# Timestamp for unique artifact directories
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

# AWS S3 Configuration - UPDATE THESE WITH YOUR ACTUAL VALUES
BUCKET_NAME = "lungsxrays01"
S3_DATA_FOLDER = "data"

# Training pipeline constants
ARTIFACT_DIR = "artifacts"
DATA_DIR = "data"
TRAIN_DIR = "train"
TEST_DIR = "test"

# Data Transformation constants
RESIZE = 224
CENTERCROP = 224
RANDOMROTATION = 10
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.1
NORMALIZE_LIST_1 = [0.485, 0.456, 0.406]
NORMALIZE_LIST_2 = [0.229, 0.224, 0.225]

# Data Loader constants
BATCH_SIZE = 32
SHUFFLE = True
PIN_MEMORY = True

# Model Training constants
EPOCH = 10
STEP_SIZE = 5
GAMMA = 0.5
DEVICE = "cuda"  # or "cpu" if no GPU

# Model Filesf.0
TRAINED_MODEL_NAME = "model.pt"
TRAIN_TRANSFORMS_FILE = "train_transforms.pt"
TEST_TRANSFORMS_FILE = "test_transforms.pt"
TRAIN_TRANSFORMS_KEY = "train_transforms"

# BentoML constants
BENTOML_MODEL_NAME = "xray_model"
BENTOML_SERVICE_NAME = "xray_service"
BENTOML_ECR_URI = "051826699665.dkr.ecr.us-east-1.amazonaws.com/xray-model"
import time

# Timestamp for unique artifact directories
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
# AWS S3 Configuration - UPDATE THESE WITH YOUR ACTUAL VALUES
BUCKET_NAME = "lungsxrays01"  # ← Change to your actual bucket name
S3_DATA_FOLDER = "data"  # ← Change to your actual folder path in S3

# Training pipeline constants
ARTIFACT_DIR = "artifacts"
DATA_DIR = "data"
TRAIN_DIR = "train"
TEST_DIR = "test"

# Data Transformation constants
RESIZE = 224
CENTERCROP = 224
RANDOMROTATION = 10
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.1
NORMALIZE_LIST_1 = [0.485, 0.456, 0.406]
NORMALIZE_LIST_2 = [0.229, 0.224, 0.225]

# Data Loader constants
BATCH_SIZE = 32
SHUFFLE = True
PIN_MEMORY = True

# Model Training constants
EPOCH = 10
STEP_SIZE = 5
GAMMA = 0.5
DEVICE = "cuda"  # or "cpu" if no GPU

# Model Files
TRAINED_MODEL_NAME = "model.pt"
TRAIN_TRANSFORMS_FILE = "train_transforms.pt"
TEST_TRANSFORMS_FILE = "test_transforms.pt"
TRAIN_TRANSFORMS_KEY = "train_transforms"

# BentoML constants
BENTOML_MODEL_NAME = "xray_model"
BENTOML_SERVICE_NAME = "xray_service"
BENTOML_ECR_URI = "051826699665.dkr.ecr.us-east-1.amazonaws.com/xray-model"