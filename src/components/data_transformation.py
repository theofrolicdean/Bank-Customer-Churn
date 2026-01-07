import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

ARTIFACTS_PATH = "artifacts/"

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(ARTIFACTS_PATH, "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self, model_type:str, non_skewed_features: list,
                                 skewed_features: list, cat_features: list):
        # noop -> no operation
        non_skewed_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()) if model_type in ["linear", "knn"] 
                                    else ("noop", "passthrough")
        ])

        skewed_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()) if model_type in ["linear", "knn"] 
                                        else ("noop", "passthrough")
        ])

        logging.info(f"Non Skewed Numerical Features: {non_skewed_features}")
        logging.info(f"Skewed Numerical Features: {skewed_features}")
        logging.info(f"Categorical Features: {cat_features}")

        if model_type == "linear":
            cat_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ])
        else:
            cat_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("non_skewed", non_skewed_transformer, non_skewed_features),
                ("skewed", skewed_transformer, skewed_features),
                ("cat", cat_transformer, cat_features)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, 
            train_path: pd.DataFrame, test_path: pd.DataFrame, model_type: str):    
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test completed")
            logging.info("Obtaining preprocessing object")

            target = "Churn_Status"
            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]
            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]
            numeric_features = list(X_train.select_dtypes(include="number").columns)
            skewed_features = [feature for feature in numeric_features
                               if abs(X_train[feature].skew()) > 0.5]
            non_skewed_features = list(set(numeric_features) - set(skewed_features)) 
            cat_features = list(X_train.select_dtypes(exclude="number").columns)
            preprocessing_obj = self.get_data_transformer_obj(
                model_type=model_type,
                non_skewed_features=non_skewed_features,
                skewed_features=skewed_features,
                cat_features=cat_features
            )

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            X_train_processed = preprocessing_obj.fit_transform(X_train)
            X_test_processed = preprocessing_obj.transform(X_test)
            train_arr = np.c_[
                X_train_processed, np.array(y_train)
            ]
            test_arr = np.c_[
                X_test_processed, np.array(y_test)
            ]

            logging.info(f"Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr, test_arr, 
                    self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as err:
            raise CustomException(err, sys)


