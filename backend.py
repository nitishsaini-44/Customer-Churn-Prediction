from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import os

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using an ANN model",
    version="1.0.0"
)

# Define Pydantic model for request body
class CustomerInfo(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Global variables for models
model = None
labelEncoder_gender = None
oneHotencoder_geo = None
scaler = None

# Load models on startup
@app.on_event("startup")
async def load_models():
    global model, labelEncoder_gender, oneHotencoder_geo, scaler
    try:
        model = load_model('models/model.h5')
        with open('models/labelEncoder_gender.pkl', 'rb') as file:
            labelEncoder_gender = pickle.load(file)
        with open('models/oneHotencoder_geo.pkl', 'rb') as file:
            oneHotencoder_geo = pickle.load(file)
        with open('models/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API. Use /predict to make predictions."}

@app.post("/predict")
def predict_churn(customer: CustomerInfo):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Prepare input data as dictionary
        input_data = customer.model_dump() # Using Pydantic V2 method
        
        # One-hot encode Geography
        geo_encoded = oneHotencoder_geo.transform([[input_data['Geography']]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=oneHotencoder_geo.get_feature_names_out(['Geography'])
        )
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Label encode Gender
        input_df['Gender'] = labelEncoder_gender.transform(input_df['Gender'])
        
        # Drop Geography and concatenate with encoded geography
        input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        churn_probability = float(prediction[0][0])
        
        return {
            "churn_probability": churn_probability,
            "prediction": "CHURN" if churn_probability > 0.5 else "STAY"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
