from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import *
from src.logger import logging
from constants import ARTIFACTS_PATH
from imblearn.pipeline import Pipeline
import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(ARTIFACTS_PATH, "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Model training initialized")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            models = {
                "Logistic Regression": {
                    "model": LogisticRegression(n_jobs=-1),
                    "type": "linear"
                },
                "KNN": {
                    "model": KNeighborsClassifier(n_jobs=-1),
                    "type": "knn"
                },
                "Decision Tree": {
                    "model": DecisionTreeClassifier(random_state=42),
                    "type": "bagging"
                },
                "Random Forest": {
                    "model": RandomForestClassifier(random_state=42, n_jobs=-1),
                    "type": "bagging"
                },
                "AdaBoost": {
                    "model": AdaBoostClassifier(random_state=42),
                    "type": "boosting"
                },
                "XGBoost": {
                    "model": XGBClassifier(random_state=42),
                    "type": "boosting"
                },
                "CatBoost": {
                    "model": CatBoostClassifier(verbose=False, 
                                                random_state=42),
                    "type": "boosting_native"
                }
            }
            param_grids = {
                "Logistic Regression": {
                    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "model__penalty": ["l2"],
                    "model__solver": ["lbfgs"],
                    "model__max_iter": [500, 1000]
                },

                "KNN": {
                    "model__n_neighbors": [3, 5, 7, 9, 11],
                    "model__weights": ["uniform", "distance"],
                    "model__metric": ["euclidean", "manhattan"]
                },

                "Decision Tree": {
                    "model__max_depth": [None, 5, 10, 20, 50],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__criterion": ["gini", "entropy"]
                },

                "Random Forest": {
                    "model__n_estimators": [100, 200, 500],
                    "model__max_depth": [None, 10, 20, 50],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", "log2"]
                },

                "AdaBoost": {
                    "model__n_estimators": [50, 100, 200],
                    "model__learning_rate": [0.01, 0.05, 0.1, 1.0]
                },

                "XGBoost": {
                    "model__n_estimators": [100, 200, 500],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0],
                    "model__gamma": [0, 0.1, 0.3]
                },

                "CatBoost": {
                    "model__iterations": [200, 500, 1000],
                    "model__depth": [4, 6, 8, 10],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__l2_leaf_reg": [1, 3, 5, 7]
                }
            }
            model_report: dict = train_and_evaluate_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test, models=models,
                param=param_grids, cv=3
            )
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]   
            best_score_threshold = 0.6
            best_model = models[best_model_name]
            if best_model_name < best_score_threshold:
                raise CustomException("No best model found")
            logging.info(f"Best model found on training: {best_model_name}, score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted = best_model.predict(X_test)
            predicted_eval = get_evaluation_report(y_test=y_test, y_pred=predicted)
            
            return predicted_eval
    
        except Exception as err:
            raise CustomException(error_message=err, error_detail=sys)
