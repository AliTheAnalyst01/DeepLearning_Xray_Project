from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_object: str
    transformed_test_object: str
    train_transforms_file: str
    test_transforms_file: str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float

@dataclass
class ModelPusherArtifact:
    bentoml_model_name: str
    bentoml_service_name: str
