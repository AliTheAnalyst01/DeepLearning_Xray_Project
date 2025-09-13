import os
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import torch

from Xray.entitiy.artifact_entity import DataTransformationArtifact
from Xray.entitiy.config_entity import DataTransformationConfig
from Xray.exception import XRayException
from Xray.logger import logging

class DataTransformation:
    def __init__(self, data_ingestion_artifact, data_transformation_config: DataTransformationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config

    def get_train_transforms(self):
        """
        Create training transforms with augmentation
        """
        try:
            train_transforms = transforms.Compose([
                transforms.Resize((self.data_transformation_config.RESIZE, self.data_transformation_config.RESIZE)),
                transforms.RandomRotation(self.data_transformation_config.RANDOMROTATION),
                transforms.ColorJitter(
                    brightness=self.data_transformation_config.color_jitter_transforms["brightness"],
                    contrast=self.data_transformation_config.color_jitter_transforms["contrast"],
                    saturation=self.data_transformation_config.color_jitter_transforms["saturation"],
                    hue=self.data_transformation_config.color_jitter_transforms["hue"]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.data_transformation_config.normalize_transforms["mean"],
                    std=self.data_transformation_config.normalize_transforms["std"]
                )
            ])
            return train_transforms
        except Exception as e:
            raise XRayException(e, sys)

    def get_test_transforms(self):
        """
        Create test transforms without augmentation
        """
        try:
            test_transforms = transforms.Compose([
                transforms.Resize((self.data_transformation_config.RESIZE, self.data_transformation_config.RESIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.data_transformation_config.normalize_transforms["mean"],
                    std=self.data_transformation_config.normalize_transforms["std"]
                )
            ])
            return test_transforms
        except Exception as e:
            raise XRayException(e, sys)

    def create_data_loaders(self, train_transforms, test_transforms):
        """
        Create PyTorch DataLoaders for training and testing
        """
        try:
            # Create datasets
            train_dataset = ImageFolder(
                root=self.data_ingestion_artifact.train_file_path,
                transform=train_transforms
            )

            test_dataset = ImageFolder(
                root=self.data_ingestion_artifact.test_file_path,
                transform=test_transforms
            )

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.data_transformation_config.data_loader_params["batch_size"],
                shuffle=self.data_transformation_config.data_loader_params["shuffle"],
                pin_memory=self.data_transformation_config.data_loader_params["pin_memory"]
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.data_transformation_config.data_loader_params["batch_size"],
                shuffle=False,  # No shuffle for testing
                pin_memory=self.data_transformation_config.data_loader_params["pin_memory"]
            )

            return train_loader, test_loader

        except Exception as e:
            raise XRayException(e, sys)

    def save_transforms(self, train_transforms, test_transforms):
        """
        Save transforms for later use in inference
        """
        try:
            # Create artifact directory
            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            # Save transforms
            torch.save(train_transforms, self.data_transformation_config.train_transforms_file)
            torch.save(test_transforms, self.data_transformation_config.test_transforms_file)

            logging.info(f"Transforms saved to {self.data_transformation_config.artifact_dir}")

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Main method to orchestrate data transformation
        """
        logging.info("Entered the initiate_data_transformation method of DataTransformation class")

        try:
            # Get transforms
            train_transforms = self.get_train_transforms()
            test_transforms = self.get_test_transforms()

            # Create DataLoaders
            train_loader, test_loader = self.create_data_loaders(train_transforms, test_transforms)

            # Save transforms
            self.save_transforms(train_transforms, test_transforms)

            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
                train_transforms_file=self.data_transformation_config.train_transforms_file,
                test_transforms_file=self.data_transformation_config.test_transforms_file
            )

            logging.info("Exited the initiate_data_transformation method of DataTransformation class")
            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
