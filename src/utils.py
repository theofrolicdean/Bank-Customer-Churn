import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score
)

def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except CustomException as err:
        raise CustomException(error_message=str(err), error_detail=sys)

def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file=file_obj)
    except CustomException as err:
        raise CustomException(error_message=str(err), error_detail=sys)

def get_evaluation_report(y_test, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1_score": f1_score(y_test, y_pred, average="macro")
    }

def train_and_evaluate_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_test: pd.DataFrame, y_test: pd.DataFrame,
    models: dict, param: dict, cv=3):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            model_param = param[list(models.keys())[i]]
            cv_k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            grid_search_cv = GridSearchCV(estimator=model,
                                          param_grid=model_param, cv=cv_k_fold,
                                          scoring="recall_macro")
            print(f"Training {model_name}, params: {model_param}\n")
            grid_search_cv.fit(X_train, y_train)
            best_model = grid_search_cv.best_estimator_
            #y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            report[model_name] = {
                "metrics": get_evaluation_report(y_test, y_test_pred),
                "model": best_model
            }

        return report

    except CustomException as err:
        raise CustomException(error_message=err, error_detail=sys)
