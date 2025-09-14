import sys
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import bentoml
import torch
from bentoml.io import Image, JSON
from Xray.entitiy.artifact_entity import ModelPusherArtifact
from Xray.entitiy.config_entity import ModelPusherConfig
from Xray.exception import XRayException
from Xray.logger import logging
from Xray.components.model_training import XRayCNN

class ModelPusher:
    """
    ModelPusher class handles the deployment of trained models using BentoML.

    This class is responsible for:
    1. Packaging the trained model with BentoML
    2. Creating a prediction service
    3. Versioning the model
    4. Preparing for deployment to cloud platforms
    """

    def __init__(self, model_pusher_config: ModelPusherConfig):
        """
        Initialize ModelPusher with configuration.

        Args:
            model_pusher_config: Configuration containing BentoML settings
        """
        self.model_pusher_config = model_pusher_config
        # Create deployment directory for storing BentoML artifacts
        self.deployment_dir = os.path.join("artifacts", "model_deployment")
        os.makedirs(self.deployment_dir, exist_ok=True)

    def _load_trained_model(self, model_path: str) -> tuple:
        """
        Load the trained PyTorch model and its metadata.

        Args:
            model_path: Path to the saved model file

        Returns:
            tuple: (model, model_metadata)
        """
        logging.info(f"Loading trained model from {model_path}")

        # Load model state dict
        model_state = torch.load(model_path, map_location='cpu')

        # Extract model and metadata
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
            # Our custom save format with metadata
            model = XRayCNN(num_classes=model_state.get('num_classes', 2))
            model.load_state_dict(model_state['model_state_dict'])
            metadata = {
                'accuracy': model_state.get('accuracy', 0.0),
                'loss': model_state.get('loss', 0.0),
                'epochs': model_state.get('epochs', 0),
                'num_classes': model_state.get('num_classes', 2)
            }
        else:
            # Direct state dict
            model = XRayCNN(num_classes=2)
            model.load_state_dict(model_state)
            metadata = {'num_classes': 2}

        model.eval()  # Set to evaluation mode
        logging.info("Model loaded successfully")
        return model, metadata

    def _create_bentoml_model(self, model, metadata: dict, model_trainer_artifact) -> str:
        """
        Create and save a BentoML model with proper packaging.

        Args:
            model: Trained PyTorch model
            metadata: Model metadata
            model_trainer_artifact: Training artifact containing model info

        Returns:
            str: BentoML model tag (name:version)
        """
        logging.info("Creating BentoML model package")

        # Generate version based on timestamp and accuracy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accuracy = model_trainer_artifact.test_accuracy
        version = f"v{timestamp}_acc{accuracy:.3f}"

        # Define the model save function for BentoML
        def model_save_func(model_obj):
            """Function to save the model for BentoML"""
            # Save the model state dict
            torch.save(model_obj.state_dict(), "model_state_dict.pt")

            # Save model architecture info
            model_info = {
                'num_classes': metadata.get('num_classes', 2),
                'model_type': 'XRayCNN',
                'input_size': (3, 224, 224),
                'class_names': ['NORMAL', 'PNEUMONIA']
            }

            with open("model_info.json", "w") as f:
                json.dump(model_info, f)

        # Define the model load function for BentoML
        def model_load_func(model_path):
            """Function to load the model for BentoML"""
            # Load model info
            with open(os.path.join(model_path, "model_info.json"), "r") as f:
                model_info = json.load(f)

            # Create model instance
            model_obj = XRayCNN(num_classes=model_info['num_classes'])

            # Load state dict
            state_dict = torch.load(
                os.path.join(model_path, "model_state_dict.pt"),
                map_location='cpu'
            )
            model_obj.load_state_dict(state_dict)
            model_obj.eval()

            return model_obj

        # Save the model using BentoML (version 1.0.25 compatible)
        try:
            # For BentoML 1.0.25, we use the older API
            bentoml_model = bentoml.pytorch.save_model(
                name=self.model_pusher_config.bentoml_model_name,
                model=model,
                metadata={
                    'accuracy': accuracy,
                    'loss': model_trainer_artifact.test_loss,
                    'timestamp': timestamp,
                    'version': version,
                    'framework': 'pytorch',
                    'task': 'image_classification'
                }
            )
        except AttributeError:
            # Fallback for even older versions
            bentoml_model = bentoml.save_model(
                name=self.model_pusher_config.bentoml_model_name,
                model=model,
                metadata={
                    'accuracy': accuracy,
                    'loss': model_trainer_artifact.test_loss,
                    'timestamp': timestamp,
                    'version': version,
                    'framework': 'pytorch',
                    'task': 'image_classification'
                }
            )

        logging.info(f"BentoML model saved: {bentoml_model.tag}")
        return str(bentoml_model.tag)

    def _create_prediction_service(self, model_tag: str, transforms_path: str) -> str:
        """
        Create a BentoML prediction service for the model.

        Args:
            model_tag: BentoML model tag
            transforms_path: Path to the saved transforms

        Returns:
            str: Path to the created service file
        """
        logging.info("Creating BentoML prediction service")

        # Load the transforms
        transforms = torch.load(transforms_path, map_location='cpu')

        # Create the service code
        service_code = f'''
import bentoml
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from bentoml.io import Image as BentoImage, JSON
import json

# Load the model
model_runner = bentoml.pytorch.get("{model_tag}").to_runner()

# Load transforms
transforms_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the service
svc = bentoml.Service("{self.model_pusher_config.bentoml_service_name}", runners=[model_runner])

@svc.api(input=BentoImage(), output=JSON())
def predict(input_image: Image.Image) -> dict:
    """
    Predict whether an X-ray image shows pneumonia or normal condition.

    Args:
        input_image: PIL Image object of the X-ray

    Returns:
        dict: Prediction results with class and confidence
    """
    # Preprocess the image
    input_tensor = transforms_pipeline(input_image).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        outputs = model_runner.run(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Convert to numpy for JSON serialization
    confidence_score = confidence.item()
    predicted_class = predicted.item()

    # Map class index to name
    class_names = ['NORMAL', 'PNEUMONIA']
    predicted_class_name = class_names[predicted_class]

    return {{
        "prediction": predicted_class_name,
        "confidence": round(confidence_score, 4),
        "class_probabilities": {{
            "NORMAL": round(probabilities[0][0].item(), 4),
            "PNEUMONIA": round(probabilities[0][1].item(), 4)
        }}
    }}
'''

        # Save the service file
        service_file_path = os.path.join(self.deployment_dir, "service.py")
        with open(service_file_path, 'w') as f:
            f.write(service_code)

        logging.info(f"Prediction service created at {service_file_path}")
        return service_file_path

    def _create_bentofile(self, service_path: str) -> str:
        """
        Create a bentofile.yaml for BentoML service configuration.

        Args:
            service_path: Path to the service file

        Returns:
            str: Path to the created bentofile.yaml
        """
        logging.info("Creating bentofile.yaml configuration")

        bentofile_content = f'''service: "{self.model_pusher_config.bentoml_service_name}"
include:
  - "{service_path}"
python:
  packages:
    - torch
    - torchvision
    - pillow
    - numpy
    - bentoml
  requirements_txt: requirements.txt

docker:
  distro: debian
  python_version: "3.9"
  cuda_version: "11.8"
  env:
    - BENTOML_HOME=/home/bentoml
'''

        bentofile_path = os.path.join(self.deployment_dir, "bentofile.yaml")
        with open(bentofile_path, 'w') as f:
            f.write(bentofile_content)

        logging.info(f"Bentofile created at {bentofile_path}")
        return bentofile_path

    def _create_requirements_file(self) -> str:
        """
        Create requirements.txt for the BentoML service.

        Returns:
            str: Path to the created requirements.txt
        """
        logging.info("Creating requirements.txt")

        requirements = '''torch>=1.12.0
torchvision>=0.13.0
pillow>=9.0.0
numpy>=1.21.0
bentoml>=1.0.0
'''

        requirements_path = os.path.join(self.deployment_dir, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write(requirements)

        logging.info(f"Requirements file created at {requirements_path}")
        return requirements_path

    def initiate_model_pusher(self, model_trainer_artifact, data_transformation_artifact) -> ModelPusherArtifact:
        """
        Main method to initiate the model pushing process.

        This method:
        1. Loads the trained model
        2. Creates BentoML model package
        3. Creates prediction service
        4. Prepares deployment artifacts

        Args:
            model_trainer_artifact: Artifact from model training
            data_transformation_artifact: Artifact from data transformation

        Returns:
            ModelPusherArtifact: Complete deployment artifact
        """
        logging.info("Entered the initiate_model_pusher method of ModelPusher class")

        try:
            # Step 1: Load the trained model
            model, metadata = self._load_trained_model(model_trainer_artifact.trained_model_file_path)

            # Step 2: Create BentoML model package
            model_tag = self._create_bentoml_model(model, metadata, model_trainer_artifact)

            # Step 3: Create prediction service
            service_path = self._create_prediction_service(
                model_tag,
                data_transformation_artifact.test_transforms_file
            )

            # Step 4: Create deployment configuration files
            bentofile_path = self._create_bentofile(service_path)
            requirements_path = self._create_requirements_file()

            # Step 5: Create deployment artifact
            model_pusher_artifact = ModelPusherArtifact(
                bentoml_model_name=self.model_pusher_config.bentoml_model_name,
                bentoml_service_name=self.model_pusher_config.bentoml_service_name,
                bentoml_model_version=model_tag.split(':')[1],  # Extract version from tag
                bentoml_model_path=model_tag,
                bentoml_service_path=service_path,
                model_accuracy=model_trainer_artifact.test_accuracy,
                model_loss=model_trainer_artifact.test_loss,
                deployment_status="READY_FOR_DEPLOYMENT"
            )

            # Step 6: Save deployment information
            deployment_info = {
                'model_tag': model_tag,
                'service_path': service_path,
                'bentofile_path': bentofile_path,
                'requirements_path': requirements_path,
                'deployment_dir': self.deployment_dir,
                'accuracy': model_trainer_artifact.test_accuracy,
                'loss': model_trainer_artifact.test_loss,
                'timestamp': datetime.now().isoformat()
            }

            info_path = os.path.join(self.deployment_dir, "deployment_info.json")
            with open(info_path, 'w') as f:
                json.dump(deployment_info, f, indent=2)

            logging.info("Model pusher completed successfully")
            logging.info(f"Deployment artifacts saved in: {self.deployment_dir}")
            logging.info(f"BentoML model tag: {model_tag}")

            return model_pusher_artifact

        except Exception as e:
            logging.error(f"Model pusher failed: {str(e)}")
            raise XRayException(e, sys)
