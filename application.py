from datetime import datetime
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        bank_join_date = pd.to_datetime(request.form["bank_join_date"])
        current_year = datetime.today().year
        tenure_years = current_year - bank_join_date.year

        data = CustomData(
            bank_quarter=bank_join_date.quarter,
            credit_score=int(request.form["credit_score"]),
            country=request.form["country"],
            gender=request.form["gender"],
            age=int(request.form["age"]),
            tenure_years=tenure_years,
            account_balance=float(request.form["balance"]),
            number_of_products=int(request.form["products"]),
            has_credit_card=request.form["has_card"],
            active_status=request.form["is_active"],
            estimated_salary_eur=float(request.form["salary"])
        )

        df = data.get_dataframe()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)[0]

        prediction_text = (
            "Customer is likely to churn"
            if prediction == 1
            else "Customer is likely to stay"
        )

    except Exception as err:
        print(err)
        prediction_text = "Unable to generate prediction. Please check inputs."

    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
