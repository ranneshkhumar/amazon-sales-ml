from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model at startup
model = joblib.load("model.pkl")

class SalesInput(BaseModel):
    order_date: str
    product_category: str
    price: float
    discount_percent: float
    quantity_sold: int
    customer_region: str
    payment_method: str
    rating: float
    review_count: int
    discounted_price: float

@app.get("/")
def home():
    return {"message": "Amazon Revenue Prediction API Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: SalesInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]

    return {
        "predicted_total_revenue": float(prediction)
    }
