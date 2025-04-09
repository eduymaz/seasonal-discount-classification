from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# FastAPI uygulaması
app = FastAPI(
    title="Seasonal Discount Effectiveness Predictor",
    description="Decision Tree tabanlı mevsimsel indirim etkisi tahminleme API'si",
    version="1.0"
)

model = joblib.load("machinelearning/model.pkl")

class DiscountFeatures(BaseModel):
    units_in_stock: float
    unit_price: float
    quantity: float
    discount: float
    total_price: float
    winsorize_unit_price: float
    discount_ratio: float
    is_discounted: int
    city_encoded: int
    product_name_encoded: int
    category_name_encoded: int
    yearquarter_encoded: int

@app.get("/")
def read_root():
    return {"message": "API çalışıyor 🚀"}

@app.post("/predict")
def predict_discount_effectiveness(features: DiscountFeatures):
    input_data = np.array([[
        features.units_in_stock,
        features.unit_price,
        features.quantity,
        features.discount,
        features.total_price,
        features.winsorize_unit_price,
        features.discount_ratio,
        features.is_discounted,
        features.city_encoded,
        features.product_name_encoded,
        features.category_name_encoded,
        features.yearquarter_encoded
    ]])

    prediction = model.predict(input_data)[0]
    return {
        "discount_effective_prediction": int(prediction),
        "message": "1 = Sales are booming! Time to scale up operations 🚀. , 0 = Sales are sluggish. Let's strategize for improvement 📉"
    }
