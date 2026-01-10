import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.constants import *
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(ARTIFACTS_PATH, "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.label_encoder = LabelEncoder()

    def transform_target(self, y):
        y_encoded = self.label_encoder.fit_transform(y)
        return pd.DataFrame(y_encoded, columns=[y.name])
        
    def get_data_transformer_obj(self, non_skewed_features: list,
                                 skewed_features: list, cat_features: list):
        # noop -> no operation
        non_skewed_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        skewed_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        logging.info(f"Non Skewed Numerical Features: {non_skewed_features}")
        logging.info(f"Skewed Numerical Features: {skewed_features}")
        logging.info(f"Categorical Features: {cat_features}")

        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("non_skewed", non_skewed_transformer, non_skewed_features),
                ("skewed", skewed_transformer, skewed_features),
                ("cat", cat_transformer, cat_features),
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, 
            train_path: str, test_path: str, use_smote=False):    
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test completed")
            logging.info("Obtaining preprocessing object")
            logging.info("Transforming training and testing target label")

            X_train = train_df.drop(columns=[TARGET_LABEL])
            y_train = self.transform_target(train_df[TARGET_LABEL])
            X_test = test_df.drop(columns=[TARGET_LABEL])
            y_test = self.transform_target(test_df[TARGET_LABEL])
            numeric_features = list(X_train.select_dtypes(include="number").columns)
            skewed_features = [feature for feature in numeric_features
                               if abs(X_train[feature].skew()) > 0.5]
            non_skewed_features = list(set(numeric_features) - set(skewed_features)) 
            cat_features = list(X_train.select_dtypes(exclude="number").columns)
            preprocessing_obj = self.get_data_transformer_obj(
                non_skewed_features=non_skewed_features,
                skewed_features=skewed_features,
                cat_features=cat_features
            )

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            preprocessing_obj.set_output(transform="pandas")
            X_train_processed = preprocessing_obj.fit_transform(X_train)
            X_test_processed = preprocessing_obj.transform(X_test)

            logging.info("Applying SMOTE to training data")
            if use_smote:
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(
                    X_train_processed, y_train
                )
                train_df = pd.concat([X_train_resampled, y_train_resampled], 
                        axis=1)
                test_df = pd.concat([X_test_processed, y_test], axis=1)
            else:
                train_df = pd.concat([X_train_processed, y_train], axis=1)
                test_df = pd.concat([X_test_processed, y_test], axis=1)

            # train_arr = np.c_[
            #     X_train_processed, np.array(y_train)
            # ]
            # test_arr = np.c_[
            #     X_test_processed, np.array(y_test)
            # ]

            logging.info(f"Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_df, test_df, 
                    self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as err:
            raise CustomException(err, sys)


