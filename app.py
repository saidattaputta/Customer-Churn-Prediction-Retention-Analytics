
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.batch_predict import predict_batch

# Required model input features
REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# --------------------------------------------------
# Load Model
# --------------------------------------------------

from src.model_loader import load_model

model = load_model()


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Page Header
# --------------------------------------------------

st.title("Customer Churn Prediction")

st.write(
    "Predict the likelihood of a customer churning "
    "and identify potential retention opportunities."
)


# --------------------------------------------------
# Business Threshold
# --------------------------------------------------

# Business-oriented threshold for retention outreach.
# This is an illustrative operating point, not a
# statistically or economically optimized threshold.

threshold = 0.4


# --------------------------------------------------
# Customer Details Form
# --------------------------------------------------

st.header("Customer Details")

col1, col2, col3 = st.columns(3)


# --------------------------------------------------
# Column 1
# --------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# --------------------------------------------------
# Column 2
# --------------------------------------------------

with col2:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


# --------------------------------------------------
# Column 3
# --------------------------------------------------

with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Predict Churn",
    type="primary"
):

    # Construct input DataFrame.
    # Feature names and ordering match model training.

    customer = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless_billing,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges

    }])


    # --------------------------------------------------
    # Calculate Churn Probability
    # --------------------------------------------------

    churn_probability = model.predict_proba(
        customer
    )[0][1]


    # --------------------------------------------------
    # Churn Classification
    # --------------------------------------------------

    churn_prediction = (
        "Yes"
        if churn_probability >= threshold
        else "No"
    )


    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.divider()

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            label="Churn Probability",
            value=f"{churn_probability:.1%}"
        )

    with result_col2:

        st.metric(
            label="Prediction Threshold",
            value=f"{threshold:.0%}"
        )


    # Churn prediction message

    if churn_prediction == "Yes":

        st.error(
            "This customer is predicted to churn."
        )

    else:

        st.success(
            "This customer is predicted not to churn."
        )


    # Probability progress bar

    st.progress(
        float(churn_probability)
    )


    # --------------------------------------------------
    # Business Insights
    # --------------------------------------------------

    st.divider()

    st.header("Business Insights")

    # Illustrative business risk categories.
    # These are not statistically calibrated classes.

    if churn_probability >= 0.7:

        st.error(
            "Risk Level: High"
        )

        st.write(
            "Suggested Action: Prioritize this customer "
            "for a retention campaign. Consider a "
            "personalized offer or proactive support."
        )

    elif churn_probability >= 0.4:

        st.warning(
            "Risk Level: Medium"
        )

        st.write(
            "Suggested Action: Monitor this customer "
            "and consider a targeted engagement campaign."
        )

    else:

        st.success(
            "Risk Level: Low"
        )

        st.write(
            "Suggested Action: Continue regular "
            "customer engagement."
        )


    # --------------------------------------------------
    # Customer Summary
    # --------------------------------------------------

    st.divider()

    st.subheader("Customer Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.write(
            f"**Contract:** {contract}"
        )

        st.write(
            f"**Tenure:** {tenure} months"
        )

    with summary_col2:

        st.write(
            f"**Internet Service:** {internet_service}"
        )

        st.write(
            f"**Payment Method:** {payment_method}"
        )

    with summary_col3:

        st.write(
            f"**Monthly Charges:** ₹{monthly_charges:.2f}"
        )

        st.write(
            f"**Total Charges:** ₹{total_charges:.2f}"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction Platform | "
    "Machine Learning, Experiment Tracking "
    "and Retention Analytics"
)

# --------------------------------------------------
# Batch Churn Prediction
# --------------------------------------------------

st.divider()

st.header("Batch Churn Prediction")

st.write(
    "Upload a CSV file containing customer records "
    "to predict churn for multiple customers at once."
)
# --------------------------------------------------
# Download Sample CSV Template
# --------------------------------------------------

st.subheader("CSV Template")

sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0
}

template_df = pd.DataFrame([sample_customer])

template_csv = template_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download CSV Template",
    data=template_csv,
    file_name="customer_churn_template.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"],
    key="batch_upload"
)

if uploaded_file is not None:

    try:
        batch_data = pd.read_csv(uploaded_file)

        # Check required columns
        missing_columns = [
            col for col in REQUIRED_FEATURES
            if col not in batch_data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        # Keep only model input features
        batch_data = batch_data[REQUIRED_FEATURES]

        # Check for missing values
        if batch_data.isnull().any().any():
            st.error(
                "The uploaded file contains missing values. "
                "Please correct them before running predictions."
            )
            st.stop()

        # Check numeric columns
        numeric_columns = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        for col in numeric_columns:
            batch_data[col] = pd.to_numeric(
                batch_data[col],
                errors="coerce"
            )

        if batch_data[numeric_columns].isnull().any().any():
            st.error(
                "Some numeric columns contain invalid values."
            )
            st.stop()

        st.success("Uploaded file passed basic validation.")

        st.subheader("Uploaded Data Preview")

        st.dataframe(
            batch_data.head(),
            use_container_width=True
        )

        if st.button(
            "Run Batch Prediction",
            key="batch_predict_button"
        ):

            batch_results = predict_batch(
                batch_data,
                threshold=threshold
            )

            st.subheader("Batch Prediction Results")

            st.dataframe(
                batch_results,
                use_container_width=True
            )

            # Summary metrics
            total_customers = len(batch_results)

            predicted_churners = (
                batch_results["Churn Prediction"] == "Yes"
            ).sum()

            churn_rate = (
                predicted_churners / total_customers
                if total_customers > 0
                else 0
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "Total Customers",
                    total_customers
                )

            with metric_col2:
                st.metric(
                    "Predicted Churners",
                    predicted_churners
                )

            with metric_col3:
                st.metric(
                    "Predicted Churn Rate",
                    f"{churn_rate:.1%}"
                )

            # Download results
            csv_results = batch_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_results,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded file: {e}"
        )
