# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Pathways", layout="wide")

# Set the title of the app with a modern design
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>⚕️ Pathways 💊</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center;'>Predict drug adoption using weighted linear regression</h4>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# Initialize session state variables
if "regression_ran" not in st.session_state:
    st.session_state.regression_ran = False
if "proceed_choice" not in st.session_state:
    st.session_state.proceed_choice = None

# Function to generate CSV template with disclaimer
def generate_csv_template():
    # (Template generation code remains the same)
    # ... [Code omitted for brevity; it's unchanged from previous versions]
    pass

# Function to run weighted linear regression
def run_weighted_linear_regression(df, feature_weights):
    # (Data processing and regression code remains the same)
    # ... [Code omitted for brevity; it's unchanged from previous versions]

    st.success("Linear Regression Results:")
    st.table(results_df)

    # Remove the extra horizontal line here and only add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Provide download options for results
    st.subheader("Step 3: Download Results")
    corr_csv = correlation_matrix.to_csv(index=True)
    results_csv = results_df.to_csv(index=False)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Correlation Matrix 📊",
            data=corr_csv,
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label="Download Regression Results 📉",
            data=results_csv,
            file_name="regression_results.csv",
            mime="text/csv",
        )

    # Update session state to indicate regression has been run
    st.session_state.regression_ran = True

# Function to run a Random Forest model
def run_random_forest(df, feature_weights):
    # (Random forest code remains the same)
    # ... [Code omitted for brevity; it's unchanged from previous versions]

# Step 1: Upload CSV
st.subheader("Step 1: Upload Your CSV File")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

# Provide template download button
csv_template = generate_csv_template()
st.download_button(
    label="Need a CSV template? Download here 📄",
    data=csv_template,
    file_name="csv_template.csv",
    mime="text/csv",
)

# Default feature weights
feature_weights = {
    # (Feature weights remain the same)
    # ... [Code omitted for brevity; it's unchanged from previous versions]
}

# Process file upload and run regression
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.info("Here is a preview of your uploaded data:")
    st.dataframe(df.head())

    st.subheader("Step 2: Run Your Weighted Linear Regression")
    if st.button("Run Linear Regression") or st.session_state.regression_ran:
        if not st.session_state.regression_ran:
            st.info("Running the regression model, please wait...")
            run_weighted_linear_regression(df, feature_weights)

        # Ask the user if they want to proceed with further analysis
        st.subheader("Question: Do you want to proceed with further analysis?")

        # Yes and No buttons, styled like download buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, proceed 🔄"):
                st.session_state.proceed_choice = "yes"
        with col2:
            if st.button("No, stop ❌"):
                st.session_state.proceed_choice = "no"

        # Handle the user's choice
        if st.session_state.proceed_choice == "yes":
            st.success(
                "Proceeding to Step 4: Running a Random Forest Machine Learning model."
            )
            run_random_forest(df, feature_weights)
        elif st.session_state.proceed_choice == "no":
            st.info(
                "Great, no further analysis will be performed on this dataset. To download the prior analyses, see the options above. To run a new analysis, please refresh the page."
            )
