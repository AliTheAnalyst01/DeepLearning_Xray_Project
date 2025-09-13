import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from Xray.entitiy.artifact_entity import ModelEvaluationArtifact
from Xray.entitiy.config_entity import ModelEvaluationConfig
from Xray.exception import XRayException
from Xray.logger import logging

class ModelEvaluation:
    """
    Model Evaluation component for assessing model performance
    """
    def __init__(self, model_evaluation_config: ModelEvaluationConfig):
        self.model_evaluation_config = model_evaluation_config

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple:
        """
        Evaluate the model on test data
        Returns: (test_loss, test_accuracy, predictions, true_labels)
        """
        logging.info("Entered the evaluate_model method of ModelEvaluation class")
        try:
            model.eval()
            test_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            all_predictions = []
            all_true_labels = []

            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(device), target.to(device)

                    # Forward pass
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    test_loss += loss.item()

                    # Get predictions
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += target.size(0)
                    correct_predictions += (predicted == target).sum().item()

                    # Store for detailed analysis
                    all_predictions.extend(predicted.cpu().numpy())
                    all_true_labels.extend(target.cpu().numpy())

                    if batch_idx % 10 == 0:
                        logging.info(f"Batch {batch_idx}/{len(test_loader)}, Loss: {loss.item():.4f}")

            # Calculate metrics
            test_loss /= len(test_loader)
            test_accuracy = 100.0 * correct_predictions / total_samples

            logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

            return test_loss, test_accuracy, all_predictions, all_true_labels

        except Exception as e:
            logging.error(f"Error in evaluate_model: {str(e)}")
            raise XRayException(e, sys)

    def generate_classification_report(self, y_true: list, y_pred: list, class_names: list = None) -> str:
        """
        Generate detailed classification report
        """
        logging.info("Entered the generate_classification_report method of ModelEvaluation class")
        try:
            if class_names is None:
                class_names = ['NORMAL', 'PNEUMONIA']

            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
            logging.info("Classification Report Generated Successfully")
            return report

        except Exception as e:
            logging.error(f"Error in generate_classification_report: {str(e)}")
            raise XRayException(e, sys)

    def plot_confusion_matrix(self, y_true: list, y_pred: list, class_names: list = None, save_path: str = None):
        """
        Plot and save confusion matrix
        """
        logging.info("Entered the plot_confusion_matrix method of ModelEvaluation class")
        try:
            if class_names is None:
                class_names = ['NORMAL', 'PNEUMONIA']

            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix - X-ray Classification')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Confusion matrix saved to {save_path}")

            plt.show()

        except Exception as e:
            logging.error(f"Error in plot_confusion_matrix: {str(e)}")
            raise XRayException(e, sys)

    def save_evaluation_metrics(self, test_loss: float, test_accuracy: float,
                               classification_report: str, save_path: str):
        """
        Save evaluation metrics to file
        """
        logging.info("Entered the save_evaluation_metrics method of ModelEvaluation class")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w') as f:
                f.write("=== MODEL EVALUATION RESULTS ===\n\n")
                f.write(f"Test Loss: {test_loss:.4f}\n")
                f.write(f"Test Accuracy: {test_accuracy:.2f}%\n\n")
                f.write("=== CLASSIFICATION REPORT ===\n")
                f.write(classification_report)

            logging.info(f"Evaluation metrics saved to {save_path}")

        except Exception as e:
            logging.error(f"Error in save_evaluation_metrics: {str(e)}")
            raise XRayException(e, sys)

    def initiate_model_evaluation(self, model: nn.Module, test_loader: DataLoader,
                                 device: torch.device) -> ModelEvaluationArtifact:
        """
        Initiate the model evaluation process
        """
        logging.info("Entered the initiate_model_evaluation method of ModelEvaluation class")
        try:
            # Evaluate model
            test_loss, test_accuracy, predictions, true_labels = self.evaluate_model(
                model, test_loader, device
            )

            # Generate classification report
            class_names = ['NORMAL', 'PNEUMONIA']
            classification_report = self.generate_classification_report(
                true_labels, predictions, class_names
            )

            # Create evaluation directory
            evaluation_dir = os.path.join("artifacts", "model_evaluation")
            os.makedirs(evaluation_dir, exist_ok=True)

            # Save evaluation metrics
            metrics_path = os.path.join(evaluation_dir, "evaluation_metrics.txt")
            self.save_evaluation_metrics(test_loss, test_accuracy, classification_report, metrics_path)

            # Plot and save confusion matrix
            confusion_matrix_path = os.path.join(evaluation_dir, "confusion_matrix.png")
            self.plot_confusion_matrix(true_labels, predictions, class_names, confusion_matrix_path)

            # Create artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                model_evaluation_path=evaluation_dir,
                test_accuracy=test_accuracy,
                test_loss=test_loss
            )

            logging.info("Model evaluation completed successfully")
            return model_evaluation_artifact

        except Exception as e:
            logging.error(f"Error in initiate_model_evaluation: {str(e)}")
            raise XRayException(e, sys)
