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
from src.components.constants import *
import os
import sys
import warnings

warnings.filterwarnings("ignore")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(ARTIFACTS_PATH, "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_df, test_df):
        try:
            logging.info("Model training initialized")
            X_train = train_df.drop(columns=[TARGET_LABEL])
            y_train = train_df[TARGET_LABEL]
            X_test = test_df.drop(columns=[TARGET_LABEL])
            y_test = test_df[TARGET_LABEL]
            models = {
                "Logistic Regression": LogisticRegression(n_jobs=-1),
                "KNN": KNeighborsClassifier(n_jobs=-1),
                # "Decision Tree": DecisionTreeClassifier(random_state=42),
                # "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                # "AdaBoost": AdaBoostClassifier(random_state=42),
                # "XGBoost": XGBClassifier(random_state=42),
                # "CatBoost": CatBoostClassifier(verbose=False, random_state=42)
            }
            param_grids = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1.0],
                    "solver": ["lbfgs"],
                    "max_iter": [2000]
                },

                "KNN": {
                    "n_neighbors": [5, 9],
                    "weights": ["uniform", "distance"]
                },

                # "Decision Tree": {
                #     "max_depth": [None, 10, 30],
                #     "min_samples_split": [2, 10],
                #     "criterion": ["gini"]
                # },

                # "Random Forest": {
                #     "n_estimators": [100, 300],
                #     "max_depth": [None, 20],
                #     "min_samples_leaf": [1, 2],
                #     "max_features": ["sqrt"]
                # },

                # "AdaBoost": {
                #     "n_estimators": [100, 200],
                #     "learning_rate": [0.05, 0.1]
                # },

                # "XGBoost": {
                #     "n_estimators": [200],
                #     "max_depth": [3, 5],
                #     "learning_rate": [0.05, 0.1],
                #     "subsample": [0.8],
                #     "colsample_bytree": [0.8]
                # },

                # "CatBoost": {
                #     "iterations": [300],
                #     "depth": [6, 8],
                #     "learning_rate": [0.05],
                #     "l2_leaf_reg": [3]
                # }
            }


            model_report: dict = train_and_evaluate_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test, models=models,
                param=param_grids, cv=3
            )
            best_model_name, best_result = max(
                model_report.items(),
                key=lambda item: item[1]["metrics"]["recall"]
            )
            best_score_threshold = 0.6
            best_model = best_result["model"]
            best_recall = best_result["metrics"]["recall"]
            if best_recall < best_score_threshold:
                raise CustomException("No best model found")

            logging.info(
                f"Best model found: {best_model_name}, recall: {best_recall}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted = best_model.predict(X_test)
            predicted_eval = get_evaluation_report(y_test=y_test, y_pred=predicted)

            return predicted_eval
    
        except Exception as err:
            raise CustomException(error_message=err, error_detail=sys)
