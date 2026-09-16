import streamlit as st
import pandas as pd

from src.predict import (
    load_model,
    load_threshold,
    predict_customer_churn
)

# Load trained model and prediction threshold
model = load_model()
threshold = load_threshold()

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("📊 Customer Churn Prediction")

st.write(
    "Predict the probability of customer churn using a trained XGBoost machine learning model."
)

st.divider()

st.subheader("Customer Information")

st.info(
    "Enter the customer's information below to generate a churn prediction."
)


# ---------------------------------------------------------
# Customer Demographics
# ---------------------------------------------------------

st.markdown("### 👤 Customer Demographics")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)


# ---------------------------------------------------------
# Service Information
# ---------------------------------------------------------

st.markdown("### 📱 Service Information")

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


# ---------------------------------------------------------
# Additional Services
# ---------------------------------------------------------

st.markdown("### 🛡️ Additional Services")

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col2:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

with col3:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


# ---------------------------------------------------------
# Account Information
# ---------------------------------------------------------

st.markdown("### 📄 Account Information")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer automatic",
            "Credit card automatic"
        ]
    )


# ---------------------------------------------------------
# Financial and Tenure Information
# ---------------------------------------------------------

st.markdown("### 💰 Financial & Tenure Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=0.01
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=0.01
    )



    # ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

st.divider()

st.subheader("🔮 Churn Prediction")

st.write(
    "Click the button below to generate the customer's churn probability."
)

if st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    # Create a DataFrame using the same feature names
    # and structure used during model training.
    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

       # Generate prediction
    churn_probability, churn_prediction = predict_customer_churn(
        model,
        customer_data,
        threshold
    )

    # Display probability
    st.metric(
        "Churn Probability",
        f"{churn_probability:.2%}"
    )

    # Display decision threshold
    st.caption(
        f"Decision threshold: {threshold:.2%}"
    )

    # Visualize churn probability
    st.progress(
        float(churn_probability),
        text=f"Churn probability: {churn_probability:.1%}"
    )

    # Display prediction
    if churn_prediction == 1:
        st.error(
            "⚠️ Prediction: Customer is likely to churn."
        )
    else:
        st.success(
            "✅ Prediction: Customer is unlikely to churn."
        )

    # Display risk level
    if churn_probability < threshold:
        risk_level = "Lower Risk"
    elif churn_probability < 0.60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Higher Risk"

    st.info(
        f"Risk Level: {risk_level}"
    )
    st.write(
        f"The model estimates a "
        f"{churn_probability:.2%} probability of churn. "
        f"The decision threshold used by the application is "
        f"{threshold:.2%}."
    )