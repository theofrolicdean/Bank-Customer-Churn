import os
import sys
import pandas as pd
from src.exception import CustomException
from src.components.constants import *
from src.utils import *

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join(ARTIFACTS_PATH, "model.pkl")
            preprocessor_path = os.path.join(ARTIFACTS_PATH, "preprocessor.pkl")
            print("Before loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After loading")
            df_scaled = preprocessor.transform(features)
            pred = model.predict(df_scaled)
            return pred 
        except Exception as err:
            raise CustomException(error_message=err, error_detail=sys)

class CustomData:
    def __init__(self, 
        bank_quarter: int,
        credit_score: int,
        country: str,
        gender: str,
        age: int,
        tenure_years: int,
        account_balance: float,
        number_of_products: int,
        has_credit_card: str,
        active_status: str,
        estimated_salary_eur: float):
        self.bank_quarter = bank_quarter
        self.credit_score = credit_score
        self.country = country
        self.gender = gender
        self.age = age
        self.tenure_years = tenure_years
        self.account_balance = account_balance
        self.number_of_products = number_of_products
        self.has_credit_card = has_credit_card
        self.active_status = active_status
        self.estimated_salary_eur = estimated_salary_eur

    def get_dataframe(self):
        try:
            input_dict = {
                "Bank_Quarter": [self.bank_quarter],
                "Credit_Score": [self.credit_score],
                "Country": [self.country],
                "Age": [self.age],
                "Tenure_Years": [self.tenure_years],
                "Account_Balance": [self.account_balance],
                "Has_Credit_Card": [self.has_credit_card],
                "Active_Status": [self.active_status],
                "Estimated_Salary_EUR": [self.estimated_salary_euf]
            }
            return pd.DataFrame(input_dict) 
        except Exception as err:
            raise CustomException(error_message=str(err), error_detail=sys)

        