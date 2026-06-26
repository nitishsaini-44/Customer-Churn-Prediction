import streamlit as st
import requests
import os

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🔮 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn or not using our FastAPI backend")

# FastAPI endpoint URL (Defaults to localhost for local testing)
API_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/predict")

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
    
    # Prepare input data matching the FastAPI CustomerInfo schema
    input_data = {
        'CreditScore': int(credit_score),
        'Geography': geography,
        'Gender': gender,
        'Age': int(age),
        'Tenure': int(tenure),
        'Balance': float(balance),
        'NumOfProducts': int(num_products),
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': float(estimated_salary)
    }
    
    with st.spinner("Connecting to FastAPI backend..."):
        try:
            response = requests.post(API_URL, json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                churn_probability = result.get("churn_probability", 0.0)
                
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
                import pandas as pd
                input_df = pd.DataFrame([input_data])
                st.dataframe(input_df.T, width='stretch')
                
            else:
                st.error(f"Error from API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Is the FastAPI server running on port 8000?")

