import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🔮 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn or not")

# Load model and preprocessors
@st.cache_resource
def load_models():
    model = load_model('model.h5')
    with open('labelEncoder_gender.pkl', 'rb') as file:
        labelEncoder_gender = pickle.load(file)
    with open('oneHotencoder_geo.pkl', 'rb') as file:
        oneHotencoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, labelEncoder_gender, oneHotencoder_geo, scaler

try:
    model, labelEncoder_gender, oneHotencoder_geo, scaler = load_models()
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Create input form
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)

with col2:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    balance = st.number_input("Balance", min_value=0.0, value=60000.0)

with col3:
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    has_credit_card = st.selectbox("Has Credit Card", [1, 0], help="1 = Yes, 0 = No")
    is_active_member = st.selectbox("Is Active Member", [1, 0], help="1 = Yes, 0 = No")

estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Prediction button
if st.button("Predict Churn", type="primary", width='stretch'):
    
    # Prepare input data
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    
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
    churn_probability = prediction[0][0]
    
    # Display results
    st.markdown("---")
    st.header("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_probability:.2%}",
            delta=None
        )
    
    with col2:
        if churn_probability > 0.5:
            st.error("⚠️ **The customer is likely to CHURN**")
            recommendation = "Consider retention strategies such as discounts or improved service."
        else:
            st.success("✅ **The customer is likely to STAY**")
            recommendation = "Customer is satisfied. Maintain current service levels."
        
        st.info(recommendation)
    
    # Display input data summary
    st.subheader("Input Summary")
    st.dataframe(input_df.T, width='stretch')
