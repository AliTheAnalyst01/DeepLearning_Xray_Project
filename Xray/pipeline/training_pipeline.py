import sys
from Xray.components.data_ingestion import DataIngestion
from Xray.components.data_transformation import DataTransformation
from Xray.components.model_training import ModelTrainer
from Xray.components.model_evaluation import ModelEvaluation
from Xray.components.model_pusher import ModelPusher
from Xray.exception import XRayException
from Xray.logger import logging
from Xray.entitiy.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)

from Xray.entitiy.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from s3 bucket")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config,
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from s3")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise XRayException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifact
        except Exception as e:
            raise XRayException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        logging.info("Entered the start_model_trainer method of TrainPipeline class")
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifact
        except Exception as e:
            raise XRayException(e, sys)

    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            # Step 1: Download data from S3
            logging.info("Starting data ingestion...")
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            logging.info("Data ingestion completed successfully")

            # Step 2: Transform data for training
            logging.info("Starting data transformation...")
            data_transformation_artifact: DataTransformationArtifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            logging.info("Data transformation completed successfully")

            # Step 3: Train the model
            logging.info("Starting model training...")
            model_trainer_artifact: ModelTrainerArtifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            logging.info("Model training completed successfully")

            # Step 4: Evaluate the model
            logging.info("Starting model evaluation...")
            model_evaluation_artifact: ModelEvaluationArtifact = self.start_model_evaluation(
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact
            )
            logging.info("Model evaluation completed successfully")

            logging.info("Pipeline completed successfully")
        except Exception as e:
            logging.error(f"Pipeline failed with error: {str(e)}")
            raise XRayException(e, sys)

    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact,
                              data_transformation_artifact: DataTransformationArtifact) -> ModelEvaluationArtifact:
        """
        Start the model evaluation process
        """
        logging.info("Entered the start_model_evaluation method of TrainPipeline class")
        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config
            )

            # Load the trained model
            import torch
            from Xray.components.model_training import XRayCNN
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model state dict
            model_state = torch.load(model_trainer_artifact.trained_model_file_path, map_location=device)
            if isinstance(model_state, dict):
                # Check if it contains model_state_dict key (our custom save format)
                if 'model_state_dict' in model_state:
                    model = XRayCNN(num_classes=model_state.get('num_classes', 2))
                    model.load_state_dict(model_state['model_state_dict'])
                else:
                    # Direct state dict
                    model = XRayCNN(num_classes=2)
                    model.load_state_dict(model_state)
            else:
                # If it's already a model
                model = model_state
            model.to(device)

            # Get test data loader
            test_loader = data_transformation_artifact.transformed_test_object

            # Evaluate the model
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation(
                model=model,
                test_loader=test_loader,
                device=device
            )

            logging.info("Model evaluation completed successfully")
            return model_evaluation_artifact

        except Exception as e:
            logging.error(f"Model evaluation failed with error: {str(e)}")
            raise XRayException(e, sys)
