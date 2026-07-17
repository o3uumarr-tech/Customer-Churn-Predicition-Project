import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("customer_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the company.")

st.divider()

# =============================
# Customer Information
# =============================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

st.divider()

# =============================
# Extra Services
# =============================

st.subheader("🌐 Internet Services")

col3, col4 = st.columns(2)

with col3:

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with col4:

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

st.divider()

# =============================
# Billing Information
# =============================

st.subheader("💳 Billing Information")

col5, col6 = st.columns(2)

with col5:

    paperless = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

with col6:

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1500.0
    )

st.divider()

predict = st.button(
    "Predict Customer Churn",
    use_container_width=True
)
# =============================
# Prepare Input Data
# =============================

if predict:

    input_data = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,

        "gender_Male": gender == "Male",

        "Partner_Yes": partner == "Yes",

        "Dependents_Yes": dependents == "Yes",

        "PhoneService_Yes": phone_service == "Yes",

        "MultipleLines_No phone service": multiple_lines == "No phone service",
        "MultipleLines_Yes": multiple_lines == "Yes",

        "InternetService_Fiber optic": internet_service == "Fiber optic",
        "InternetService_No": internet_service == "No",

        "OnlineSecurity_No internet service": online_security == "No internet service",
        "OnlineSecurity_Yes": online_security == "Yes",

        "OnlineBackup_No internet service": online_backup == "No internet service",
        "OnlineBackup_Yes": online_backup == "Yes",

        "DeviceProtection_No internet service": device_protection == "No internet service",
        "DeviceProtection_Yes": device_protection == "Yes",

        "TechSupport_No internet service": tech_support == "No internet service",
        "TechSupport_Yes": tech_support == "Yes",

        "StreamingTV_No internet service": streaming_tv == "No internet service",
        "StreamingTV_Yes": streaming_tv == "Yes",

        "StreamingMovies_No internet service": streaming_movies == "No internet service",
        "StreamingMovies_Yes": streaming_movies == "Yes",

        "Contract_One year": contract == "One year",
        "Contract_Two year": contract == "Two year",

        "PaperlessBilling_Yes": paperless == "Yes",

        "PaymentMethod_Credit card (automatic)": payment == "Credit card (automatic)",
        "PaymentMethod_Electronic check": payment == "Electronic check",
        "PaymentMethod_Mailed check": payment == "Mailed check"
    }

    input_df = pd.DataFrame([input_data])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]
        # =============================
    # Display Result
    # =============================

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")

    else:
        st.success("✅ This customer is likely to stay.")

    st.write(f"### Churn Probability: {probability:.2%}")

    st.progress(float(probability))

    st.divider()

    st.subheader("Customer Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Gender",
            "Senior Citizen",
            "Partner",
            "Dependents",
            "Tenure",
            "Phone Service",
            "Internet Service",
            "Contract",
            "Monthly Charges",
            "Total Charges",
            "Payment Method"
        ],

        "Value": [
            gender,
            senior,
            partner,
            dependents,
            tenure,
            phone_service,
            internet_service,
            contract,
            monthly,
            total,
            payment
        ]
    })

    st.dataframe(summary, use_container_width=True)

    st.divider()

    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "Random Forest")

    with col2:
        st.metric("Features", "30")

    with col3:
        st.metric("Prediction", "Churn" if prediction == 1 else "Stay")

st.divider()

st.caption("Made by Omar El-Zoghby | Machine Learning Project | Streamlit")