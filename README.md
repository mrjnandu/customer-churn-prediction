# Customer Churn Prediction

## Project Overview
This project uses Machine Learning to predict whether a telecom customer is likely to leave the company (customer churn).

## Features
- Predicts customer churn using customer details.
- Uses Logistic Regression for classification.
- Includes data preprocessing and feature encoding.
- Provides an interactive web interface using Streamlit.
- Displays estimated churn probability and prediction results.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib and Seaborn (for exploratory data analysis)

## Machine Learning Workflow
1. Load and explore the dataset.
2. Clean missing and invalid values.
3. Preprocess numerical and categorical features.
4. Split data into training and testing sets.
5. Train a Logistic Regression model.
6. Evaluate the model.
7. Save the trained model and deploy the application.

## Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Dataset
Telco Customer Churn dataset.

## Disclaimer
Predictions are estimates from a machine learning model and are not guaranteed outcomes.
