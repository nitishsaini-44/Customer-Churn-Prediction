# 🔮 Customer Churn Prediction

An Artificial Neural Network (ANN) based web application to predict whether a customer is likely to churn or continue their relationship with a company.

## 📋 Project Overview

This project builds and deploys a machine learning model that predicts customer churn using an ANN built with TensorFlow/Keras. The model is wrapped in an interactive Streamlit web application for easy prediction and visualization.

### Dataset
- **File**: `Churn_Modelling.csv`
- **Features**: Customer demographics, account information, and behavioral metrics
- **Target**: Churn (0 = No churn, 1 = Churn)

## 🏗️ Project Structure

```
ANNClassification/
├── app.py                          # Streamlit web application
├── prediction.ipynb               # Notebook for making predictions
├── EDA.ipynb                      # Exploratory Data Analysis
├── model.h5                       # Trained neural network model
├── Churn_Modelling.csv            # Dataset
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── labelEncoder_gender.pkl        # Preprocessor for gender encoding
├── oneHotencoder_geo.pkl          # Preprocessor for geography encoding
├── scaler.pkl                     # StandardScaler for feature normalization
└── logs/                          # TensorFlow training logs
    └── fit/                       # Training history
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download the project**
   ```bash
   cd ANNClassification
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📱 Running the Application

Start the Streamlit web app:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the App

1. **Enter Customer Information**:
   - Credit Score
   - Geography (France, Germany, Spain)
   - Gender (Male, Female)
   - Age
   - Tenure (years with bank)
   - Balance
   - Number of Products
   - Credit Card status
   - Active Member status
   - Estimated Salary

2. **Click "Predict Churn"** button

3. **View Results**:
   - Churn probability percentage
   - Risk assessment (Likely to Churn or Likely to Stay)
   - Actionable recommendation
   - Input data summary

## 📊 Model Details

- **Architecture**: Deep Artificial Neural Network
- **Input Features**: 12 (after preprocessing)
- **Framework**: TensorFlow/Keras
- **Output**: Binary classification (Churn probability 0-1)

### Preprocessing Pipeline
1. **Label Encoding**: Gender (Male/Female → 0/1)
2. **One-Hot Encoding**: Geography (France, Germany, Spain)
3. **StandardScaling**: Feature normalization using fitted scaler

## 📓 Notebooks

### `EDA.ipynb`
- Exploratory Data Analysis
- Data visualization and statistical insights
- Feature distribution and correlations

### `prediction.ipynb`
- Step-by-step prediction pipeline
- Model loading and preprocessing
- Example prediction workflow

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web application framework |
| tensorflow | 2.15.0 | Deep learning framework |
| keras | 2.15.0 | Neural network API |
| pandas | 2.1.3 | Data manipulation |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.2 | Preprocessing and utilities |
| tensorboard | 2.15.0 | Training visualization |
| matplotlib | 3.8.2 | Data visualization |
| scikeras | 0.13.0 | Scikit-learn wrapper for Keras |

## 🔧 Model Training

To retrain the model (if needed):
1. Use the training notebook or script
2. Save the new model as `model.h5`
3. Update the preprocessor pickle files if training data changes
4. Restart the Streamlit app

## 📈 Model Performance

The model is evaluated on:
- **Accuracy**: Classification accuracy on test data
- **Precision/Recall**: For handling class imbalance
- **AUC-ROC**: Area under the ROC curve

Check the TensorFlow logs in the `logs/` directory for detailed training metrics.

## 🎯 Use Cases

- **Customer Retention**: Identify at-risk customers
- **Targeted Interventions**: Focus retention efforts on high-churn-probability customers
- **Risk Assessment**: Evaluate customer lifetime value
- **Business Intelligence**: Understand churn drivers

## ⚠️ Important Notes

- The model predicts churn probability; threshold is set at **0.5**
- Predictions are based on the input features provided
- For best results, ensure input values are realistic ranges
- Regularly retrain the model with new customer data

## 🤝 Contributing

Feel free to improve the model, add new features, or enhance the UI.

## 📄 License

This project is open for educational and commercial use.

## 📧 Contact & Support

For issues or questions, refer to the project documentation or notebooks.

---

**Made with ❤️ for data science & machine learning**
