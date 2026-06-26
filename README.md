# 🔮 Customer Churn Prediction

An Artificial Neural Network (ANN) based web application to predict whether a customer is likely to churn or continue their relationship with a company.

## 📋 Project Overview 

This project builds and deploys a machine learning model that predicts customer churn using an ANN built with TensorFlow/Keras. The application is built using a modern **microservices architecture**:
1. **Backend**: A high-performance REST API built with **FastAPI** that handles the heavy lifting of model inference and data preprocessing.
2. **Frontend**: An interactive web application built with **Streamlit** that consumes the backend API for easy prediction and visualization.

### Dataset
- **File**: `data/Churn_Modelling.csv`
- **Features**: Customer demographics, account information, and behavioral metrics
- **Target**: Churn (0 = No churn, 1 = Churn)

## 🏗️ Project Structure

```
ANNClassification/
├── app.py                          # Streamlit Frontend application
├── backend.py                      # FastAPI Backend server
├── Procfile                        # Configuration for deploying backend to Render
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── data/
│   └── Churn_Modelling.csv         # Dataset
├── models/
│   ├── model.h5                    # Trained neural network model
│   ├── labelEncoder_gender.pkl     # Preprocessor for gender encoding
│   ├── oneHotencoder_geo.pkl       # Preprocessor for geography encoding
│   └── scaler.pkl                  # StandardScaler for feature normalization
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   ├── prediction.ipynb            # Notebook for testing predictions
│   └── hyperparatuningann.ipynb    # Hyperparameter tuning for ANN
└── logs/                           # TensorFlow training logs
```

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Anaconda (recommended for managing TensorFlow dependencies)

### Installation

1. **Clone or download the project**
   ```bash
   cd ANNClassification
   ```

2. **Create/Activate an environment and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📱 Running the Application (Local Development)

Since this project separates the frontend and backend, you will need to run two terminal windows simultaneously.

**Terminal 1: Start the FastAPI Backend**
```bash
python -m uvicorn backend:app --port 8000
```
*The backend API will run at `http://127.0.0.1:8000`. You can view the interactive API documentation at `http://127.0.0.1:8000/docs`.*

**Terminal 2: Start the Streamlit Frontend**
```bash
streamlit run app.py
```
*The application UI will open in your browser at `http://localhost:8501`. It will automatically connect to your local backend API.*

## ☁️ Deployment Guide

This application is designed to be deployed across two distinct services for maximum efficiency: **Render** (for the heavy machine learning backend) and **Streamlit Community Cloud** (for the lightweight frontend).

### Step 1: Push Code to GitHub
Ensure all your files are committed and pushed to a GitHub repository. Both Render and Streamlit Cloud will pull your code directly from GitHub.

### Step 2: Deploy the Backend to Render
1. Create a free account on [Render](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Configure the Web Service:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend:app --host 0.0.0.0 --port $PORT` (this is automatically handled by the included `Procfile`).
   - **Instance Type:** `Free` tier.
5. Click **Create Web Service**.
6. Wait for the build to finish. *(Note: TensorFlow is a heavy dependency and may take a few minutes to install)*.
7. Once successful, copy the public URL provided by Render (e.g., `https://your-api.onrender.com`).

### Step 3: Deploy the Frontend to Streamlit Cloud
1. Go to [Streamlit Community Cloud](https://share.streamlit.io) and log in.
2. Click **New app** -> **Use existing repo**.
3. Select your repository and set the **Main file path** to `app.py`.
4. **Important Configuration:** Before deploying, click on **Advanced settings**.
5. In the **Secrets** section, configure the frontend to talk to your live backend by adding your Render URL:
   ```toml
   BACKEND_URL = "https://your-api.onrender.com/predict"
   ```
6. Save the settings and click **Deploy**.

Your Streamlit application is now live and securely querying your deployed FastAPI backend!

## 📊 Model Details

- **Architecture**: Deep Artificial Neural Network
- **Input Features**: 12 (after preprocessing)
- **Framework**: TensorFlow/Keras
- **Output**: Binary classification (Churn probability 0-1)

### Preprocessing Pipeline
1. **Label Encoding**: Gender (Male/Female → 0/1)
2. **One-Hot Encoding**: Geography (France, Germany, Spain)
3. **StandardScaling**: Feature normalization using fitted scaler

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Frontend web application framework |
| fastapi | High-performance backend API framework |
| uvicorn | ASGI server to run FastAPI |
| pydantic | Data validation for the API |
| tensorflow | Deep learning framework |
| pandas / numpy | Data manipulation and numerical computing |
| scikit-learn | Data preprocessing and scaling |

## 🤝 Contributing

Feel free to improve the model, add new features, or enhance the UI.

## 📄 License

This project is open for educational and commercial use.
