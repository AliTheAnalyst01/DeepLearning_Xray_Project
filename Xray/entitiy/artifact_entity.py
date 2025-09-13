from dataclasses import dataclass
from torch.utils.data import DataLoader

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_object: DataLoader
    transformed_test_object: DataLoader
    train_transforms_file: str
    test_transforms_file: str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float

@dataclass
class ModelEvaluationArtifact:
    model_evaluation_path: str
    test_accuracy: float
    test_loss: float

@dataclass
class ModelPusherArtifact:
    bentoml_model_name: str
    bentoml_service_name: str
