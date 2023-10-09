from src.MLproject.logger import logging
from src.MLproject.exception import CustonException
import sys
from src.MLproject.components.data_ingestion import DataIngestion
from src.MLproject.components.data_ingestion import DataIngestionConfig
from src.MLproject.components.data_transformation import DataTransformationConfig , DataTransformation
from src.MLproject.components.model_trainer import ModelTrainerConfig , ModelTrainer


if __name__ == "__main__":
    logging.info("The execution has started")
    
    try:
    
        data_ingestion = DataIngestion()
        train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr , test_arr , preprocessor_file_path = data_transformation.initiate_data_transformation(train_data_path , test_data_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_training(train_arr , test_arr))
        
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustonException(e ,sys)