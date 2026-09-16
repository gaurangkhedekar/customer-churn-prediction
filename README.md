# Customer Churn Prediction

An end-to-end machine learning project that predicts the probability of customer churn using customer demographics, services, account information, tenure, and billing characteristics.

The project includes data exploration, preprocessing, exploratory data analysis, model comparison, hyperparameter tuning, probability-threshold optimization, model interpretation, application testing, and a Streamlit prediction interface.

---

## Project Overview

Customer churn refers to customers discontinuing a service.

The objective of this project is to build a machine learning classification system that estimates the probability that a customer will churn and converts that probability into a churn/no-churn prediction using an optimized classification threshold.

The project uses the IBM Telco Customer Churn sample dataset.

---

## Objectives

- Understand and analyze customer churn patterns.
- Perform data cleaning and preprocessing.
- Explore relationships between customer characteristics and churn.
- Prepare numerical and categorical features for machine learning.
- Compare multiple classification algorithms.
- Evaluate models using cross-validation.
- Tune model hyperparameters.
- Select a final XGBoost model.
- Optimize the probability classification threshold using F1-score.
- Analyze model feature importance.
- Build a reusable prediction module.
- Develop a Streamlit web application.
- Test the application using multiple customer scenarios.
- Organize the complete project for reproducibility and GitHub.

---

## Dataset

The project uses the IBM Telco Customer Churn sample dataset.

### Dataset characteristics

- 7,043 customer records
- 21 original columns
- Binary target variable: `Churn`
- Target classes:
  - `No`
  - `Yes`

### Important variables

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

The `customerID` column was removed before model training because it is an identifier rather than a predictive customer characteristic.

---

## Project Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Preparation
     ↓
Train/Test Split
     ↓
Preprocessing Pipeline
     ↓
Baseline Model Comparison
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Final Model Evaluation
     ↓
Feature Importance Analysis
     ↓
Threshold Optimization
     ↓
Model & Threshold Saving
     ↓
Prediction Module
     ↓
Streamlit Application
     ↓
Application Testing