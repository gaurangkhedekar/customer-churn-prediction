"""
Customer Churn Prediction Utilities

This module loads the trained XGBoost pipeline and prediction
threshold, then provides a reusable function for churn prediction.
"""

import json
from pathlib import Path

import joblib
import pandas as pd


# Determine the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model and threshold paths
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_churn_pipeline.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold_config.json"


def load_model():
    """
    Load the trained XGBoost preprocessing + model pipeline.
    """
    return joblib.load(MODEL_PATH)


def load_threshold():
    """
    Load the saved classification threshold.
    """
    with open(THRESHOLD_PATH, "r") as file:
        config = json.load(file)

    return config["threshold"]


def predict_customer_churn(model, customer_data, threshold):
    """
    Predict customer churn probability and classification.

    Parameters
    ----------
    model : fitted model pipeline
        Saved preprocessing + XGBoost pipeline.

    customer_data : pandas.DataFrame
        Customer information in the expected feature format.

    threshold : float
        Probability threshold used for classification.

    Returns
    -------
    churn_probability : float
        Predicted probability of churn.

    churn_prediction : int
        1 for Churn, 0 for No Churn.
    """

    churn_probability = model.predict_proba(
        customer_data
    )[:, 1][0]

    churn_prediction = int(
        churn_probability >= threshold
    )

    return churn_probability, churn_prediction