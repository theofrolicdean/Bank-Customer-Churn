import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score
)

def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except CustomException as err:
        raise CustomException(error_message=err, error_detail=sys)

def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file=file_obj)
    except CustomException as err:
        raise CustomException(error_message=err, error_detail=sys)

def train_and_evaluate_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_test: pd.DataFrame, y_test: pd.DataFrame,
    models: dict, param, cv=3):
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(model.keys())[i]
            model = list(model.values())[i]
            model_param = param[list(model.keys())[i]]
            grid_search_cv = GridSearchCV(estimator=model,
                                          param_grid=model_param, cv=cv,
                                          scoring="recall")
            grid_search_cv.fit(X_train, y_train)
            model.set_params(**grid_search_cv.best_params_)
            model.fit(X_train, y_train)
            #y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            report[model_name] = {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred, average="macro"),
                "recall": recall_score(y_test, y_test_pred, average="macro"),
                "f1_score": f1_score(y_test, y_test_pred, average="macro")
            }

        return report

    except CustomException as err:
        raise CustomException(error_message=err, error_detail=sys)
