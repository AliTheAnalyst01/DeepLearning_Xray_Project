from functools import total_ordering
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np

from PIL import Image
from Xray.entitiy.artifact_entity import ModelTrainerArtifact
from Xray.entitiy.config_entity import ModelTrainerConfig
from Xray.exception import XRayException
from Xray.logger import logging

class XRayCNN(nn.Module):
    """
    Custom CNN model for X_ray classification
    This is a simple but effective architecture for medical image classification
    """
    def __init__(self, num_classes=2):
        super(XRayCNN, self).__init__()

        # Enhanced convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # pooling layers - reduces image size while retaining important features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # dropout - prevents overfitting by randomly dropping out some neurons to 0
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

        # Enhanced fully connected layers
        # Assuming input image size is 224x224: 224->112->56->28->14->7
        self.fc1 = nn.Linear(512*7*7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass through the model
        """
        # Enhanced conv blocks: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224*224 -> 112*112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112*112 -> 56*56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56*56 -> 28*28
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 28*28 -> 14*14

        # Additional pooling for better feature extraction
        x = self.pool(x)  # 14*14 -> 7*7

        x = x.view(-1, 512*7*7)  # flatten the output
        x = self.dropout1(x)  # apply dropout
        x = F.relu(self.fc1(x))  # fully connected layer 1
        x = self.dropout2(x)  # apply dropout
        x = F.relu(self.fc2(x))  # fully connected layer 2
        x = self.fc3(x)  # output layer
        return x




class ModelTrainer:
    def __init__(self, data_transformation_artifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def create_model(self):
        """
        create and initialize the model
        """
        try:
            logging.info("Entered the create_model method of ModelTrainer class")
            model = XRayCNN(num_classes=2)
            model.to(self.device) # move the model to the device gpu
            logging.info("Created the model")
            return model
        except Exception as e:
            raise XRayException(e, sys)

    def create_optimizer(self,model):
        """
        creates optimizer for training - this updates mdoel weights
        """
        try:
            logging.info("Entered the create_optimizer method of ModelTrainer class")
            optimizer = optim.Adam(model.parameters(), lr=self.model_trainer_config.optimizer_params["lr"])
            logging.info("Created the optimizer")
            logging.info(f"Optimizer: {optimizer}")
            return optimizer
        except Exception as e:
            raise XRayException(e, sys)


    def create_loss_function(self):
        """
        creates loss function for training - this measures how far the model's predictions are from the actual labels
        """
        try:
            logging.info("Entered the create_loss_function method of ModelTrainer class")
            # Calculate class weights to handle imbalanced dataset
            # Weight pneumonia class more heavily to reduce false negatives
            class_weights = torch.tensor([1.0, 3.0])  # [NORMAL, PNEUMONIA] - pneumonia gets 3x weight
            if torch.cuda.is_available():
                class_weights = class_weights.cuda()

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logging.info("Created the loss function with class weights")
            logging.info(f"Loss function: {criterion}")
            logging.info(f"Class weights: {class_weights}")
            return criterion
        except Exception as e:
            raise XRayException(e, sys)

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """
        trains the model for one epoch
        """
        try:
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                # move the images and labels to the device
                images,labels = images.to(self.device),labels.to(self.device)
                # zero the gradients
                optimizer.zero_grad()
                # forward pass- get predictions
                outputs = model(images)
                # calculate the loss
                loss = criterion(outputs, labels)
                # backward pass - calculate the gradients
                loss.backward()
                # update the weights
                optimizer.step()
                # update the running loss
                running_loss += loss.item()
                # update the correct predictions
                _,predicted = torch.max(outputs, 1)
                # update the total samples
                total_samples += labels.size(0)
                # update the correct predictions
                correct_predictions += (predicted == labels).sum().item()
                # log progress every 10 batches
                if batch_idx % 10 == 0:
                    logging.info(f"Epoch [{epoch+1}/{self.model_trainer_config.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}, Accuracy: {100*correct_predictions/total_samples:.2f}%")

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100.0*correct_predictions / total_samples
            return epoch_loss, epoch_accuracy
        except Exception as e:
            raise XRayException(e, sys)

    def validate_epoch(self, model, test_loader, criterion):
        """
        validates the model for test
        """
        try:
            model.eval()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    _,predicted = torch.max(outputs, 1)
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
            epoch_loss = running_loss / len(test_loader)
            epoch_accuracy = 100.0*correct_predictions / total_samples
            return epoch_loss, epoch_accuracy
        except Exception as e:
            raise XRayException(e, sys)

    def save_model(self, model, path):
        """
        saves the model to the given path
        """
        try:
            logging.info("Entered the save_model method of ModelTrainer class")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({'model_state_dict':model.state_dict(),'model_architecture':"XRayCNN",'num_classes':2},path)
            logging.info(f"Model saved to {path}")
        except Exception as e:
            raise XRayException(e, sys)



    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered the initiate_model_trainer method of ModelTrainer class")
        try:
            train_loader = self.data_transformation_artifact.transformed_train_object
            test_loader = self.data_transformation_artifact.transformed_test_object
            model = self.create_model()
            optimizer = self.create_optimizer(model)
            criterion = self.create_loss_function()
            best_accuracy = 0.0
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            logging.info(f"Starting training for {self.model_trainer_config.epochs} epochs...")

            for epoch in range(self.model_trainer_config.epochs):
                logging.info(f"Epoch {epoch + 1}/{self.model_trainer_config.epochs}")

                # Train for one epoch
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)

                # Validate for one epoch
                val_loss, val_acc = self.validate_epoch(model, test_loader, criterion)

                # Store metrics
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                # Log epoch results
                logging.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                logging.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    self.save_model(model, self.model_trainer_config.trained_model_path)
                    logging.info(f"New best model saved with accuracy: {best_accuracy:.2f}%")

            # Final evaluation
            logging.info(f"Training completed! Best validation accuracy: {best_accuracy:.2f}%")

            # Create artifact with results
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_path,
                train_accuracy=best_accuracy,
                test_accuracy=val_acc,
                train_loss=train_losses[-1],
                test_loss=val_losses[-1]
            )

            logging.info("Exited the initiate_model_trainer method of ModelTrainer class")
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise XRayException(e, sys)
