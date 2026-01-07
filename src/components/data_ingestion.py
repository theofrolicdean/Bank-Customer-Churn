import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

ARTIFACTS_PATH = "artifacts/"
DATA_PATH = "notebook/datasets/bank_churn_combined.csv"

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(ARTIFACTS_PATH, "train.csv")
    test_data_path: str = os.path.join(ARTIFACTS_PATH, "test.csv")
    raw_data_path: str = os.path.join(ARTIFACTS_PATH, "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initialized.")
        try:
            df = pd.read_csv(DATA_PATH)
            logging.info("Read the dataset as dataframe.")
            os.makedirs(ARTIFACTS_PATH, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initialized.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, 
                             header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, 
                            header=True)
            logging.info("Data ingestion process is completed.")

            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path)
        except Exception as err:
            raise CustomException(err, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()