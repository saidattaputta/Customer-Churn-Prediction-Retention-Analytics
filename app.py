
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from src.batch_predict import predict_batch
from src.model_loader import load_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "telco_churn_clean.csv"
)


# --------------------------------------------------
# Required Model Input Features
# --------------------------------------------------

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
    "TotalCharges",
]


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = load_model()


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
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

# Illustrative business-oriented threshold.
# This is not a statistically or economically
# optimized operating point.

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
        ["Male", "Female"],
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"],
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
    )

    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12,
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )


# --------------------------------------------------
# Column 2
# --------------------------------------------------

with col2:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service",
        ],
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No",
        ],
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )


# --------------------------------------------------
# Column 3
# --------------------------------------------------

with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service",
        ],
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0,
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0,
    )


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button(
    "Predict Churn",
    type="primary",
):

    # --------------------------------------------------
    # Prepare Customer Data
    # --------------------------------------------------

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
        "TotalCharges": total_charges,
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
            value=f"{churn_probability:.1%}",
        )

    with result_col2:

        st.metric(
            label="Prediction Threshold",
            value=f"{threshold:.0%}",
        )


    # --------------------------------------------------
    # Churn Prediction Message
    # --------------------------------------------------

    if churn_prediction == "Yes":

        st.error(
            "This customer is predicted to churn."
        )

    else:

        st.success(
            "This customer is predicted not to churn."
        )


    # --------------------------------------------------
    # SHAP Explainability
    # --------------------------------------------------

    with st.expander(
        "🔍 Why did the model make this prediction?"
    ):

        try:

            # Get pipeline components
            preprocessor = model.named_steps[
                "preprocessor"
            ]

            classifier = model.named_steps[
                "classifier"
            ]


            # Transform customer data
            transformed_customer = (
                preprocessor.transform(customer)
            )


            # Load representative background data
            background_data = pd.read_csv(
                DATA_PATH
            )

            background_data = (
                background_data[
                    REQUIRED_FEATURES
                ]
                .sample(
                    n=min(
                        100,
                        len(background_data),
                    ),
                    random_state=42,
                )
            )


            # Transform background data
            transformed_background = (
                preprocessor.transform(
                    background_data
                )
            )


            # Create SHAP explainer
            explainer = shap.LinearExplainer(
                classifier,
                transformed_background,
            )


            # Explain this customer's prediction
            shap_values = explainer(
                transformed_customer
            )


            # Get transformed feature names
            transformed_feature_names = (
                preprocessor.get_feature_names_out()
            )


            # Get individual SHAP contributions
            customer_shap_values = (
                shap_values.values[0]
            )


            # Prepare feature contribution dataframe
            shap_df = pd.DataFrame({
                "Feature": (
                    transformed_feature_names
                ),
                "SHAP Value": (
                    customer_shap_values
                ),
            })


            shap_df["Absolute SHAP"] = (
                shap_df["SHAP Value"].abs()
            )


            # Get top 10 contributing features
            shap_df = (
                shap_df.sort_values(
                    "Absolute SHAP",
                    ascending=False,
                )
                .head(10)
            )


            st.write(
                "Features with the strongest influence "
                "on this customer's prediction:"
            )


            # Display contribution table
            st.dataframe(
                shap_df[
                    [
                        "Feature",
                        "SHAP Value",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


            # --------------------------------------------------
            # SHAP Feature Contribution Chart
            # --------------------------------------------------

            chart_df = shap_df.sort_values(
                "SHAP Value",
                ascending=True,
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )


            ax.barh(
                chart_df["Feature"],
                chart_df["SHAP Value"],
            )


            ax.axvline(
                x=0,
                linestyle="--",
                linewidth=1,
            )


            ax.set_xlabel(
                "SHAP Value "
                "(positive = higher churn tendency)"
            )


            ax.set_title(
                "Top Feature Contributions"
            )


            plt.tight_layout()


            st.pyplot(fig)

            plt.close(fig)


            st.caption(
                "SHAP values show how each feature "
                "influences the model's output relative "
                "to the background dataset. They explain "
                "the model, not causality."
            )


        except Exception as e:

            st.warning(
                "SHAP explanation could not be generated."
            )

            st.exception(e)


    # --------------------------------------------------
    # Probability Progress Bar
    # --------------------------------------------------

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
            "Suggested Action: Maintain current "
            "engagement and service quality."
        )


# --------------------------------------------------
# Batch Churn Prediction
# --------------------------------------------------

st.divider()

st.header("Batch Churn Prediction")

st.write(
    "Upload a CSV file containing multiple customers "
    "to generate churn predictions in bulk."
)


uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"],
)


if uploaded_file is not None:

    batch_data = pd.read_csv(
        uploaded_file
    )


    # --------------------------------------------------
    # Validate Required Features
    # --------------------------------------------------

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in batch_data.columns
    ]


    if missing_features:

        st.error(
            "The uploaded file is missing required "
            f"features: {missing_features}"
        )

    else:

        st.write(
            f"Uploaded {len(batch_data)} customers."
        )


        # --------------------------------------------------
        # Batch Prediction Threshold
        # --------------------------------------------------

        batch_threshold = st.slider(
            "Batch Prediction Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
        )


        # --------------------------------------------------
        # Generate Predictions
        # --------------------------------------------------

        if st.button(
            "Generate Batch Predictions"
        ):

            try:

                batch_results = predict_batch(
                    batch_data[
                        REQUIRED_FEATURES
                    ],
                    threshold=batch_threshold,
                )


                # --------------------------------------------------
                # Display Results
                # --------------------------------------------------

                st.subheader(
                    "Batch Prediction Results"
                )


                st.dataframe(
                    batch_results,
                    use_container_width=True,
                )


                # --------------------------------------------------
                # Summary Metrics
                # --------------------------------------------------

                total_customers = (
                    len(batch_results)
                )


                predicted_churners = (
                    batch_results[
                        "Churn Prediction"
                    ] == "Yes"
                ).sum()


                churn_rate = (
                    predicted_churners
                    / total_customers
                    if total_customers > 0
                    else 0
                )


                metric_col1, metric_col2, metric_col3 = (
                    st.columns(3)
                )


                with metric_col1:

                    st.metric(
                        "Total Customers",
                        total_customers,
                    )


                with metric_col2:

                    st.metric(
                        "Predicted Churners",
                        int(predicted_churners),
                    )


                with metric_col3:

                    st.metric(
                        "Predicted Churn Rate",
                        f"{churn_rate:.1%}",
                    )


                # --------------------------------------------------
                # Download Predictions
                # --------------------------------------------------

                csv = batch_results.to_csv(
                    index=False
                ).encode("utf-8")


                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name=(
                        "customer_churn_predictions.csv"
                    ),
                    mime="text/csv",
                )


            except Exception as e:

                st.error(
                    "Batch prediction failed."
                )

                st.exception(e)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Prediction Platform | "
    "Machine Learning, Analytics, and Explainability"
)
