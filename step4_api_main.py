import pandas as pd
import numpy as np

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import random

df = pd.read_csv('./data/processed_data.csv')

categorical_columns = ['product_name', 'category_name', 'yearquarter', 'city']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop(columns=['discount_effective', 'order_date', 'customer_id', 'product_id', 'category_id'])
y = df['discount_effective']

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_balanced, y_balanced)

#joblib.dump(model, './results/sales_model.pkl')
#joblib.dump(X.columns.tolist(), './results/model_features.pkl')


app = FastAPI(title="Sales Discount Effect Predictor", description="Predicts if discount increased sales.")

model = joblib.load('./results/sales_model.pkl')
model_columns = joblib.load('./results/model_features.pkl')

class SaleInput(BaseModel):
    order_date: str
    customer_id: str
    city: str
    product_id: int
    product_name: str
    units_in_stock: int
    unit_price: float
    quantity: int
    discount: float
    category_id: int
    category_name: str
    winsorize_unit_price: float
    total_price: float
    year: int
    yearquarter: str

def preprocess_input(sale: SaleInput):
    df_input = pd.DataFrame([sale.dict()])

    categorical_columns = ['product_name', 'category_name', 'yearquarter', 'city']
    df_input = pd.get_dummies(df_input, columns=categorical_columns, drop_first=True)

    for col in model_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    return df_input[model_columns]

@app.post("/predict")
def predict(sale: SaleInput):
    input_df = preprocess_input(sale)
    prediction = model.predict(input_df)[0]
    result = "Sales are booming! Time to scale up operations 🚀" if prediction == 1 else "Sales are sluggish. Let's strategize for improvement 📉"


    return {
        "prediction": result,
        "input_summary": sale.dict()
    }