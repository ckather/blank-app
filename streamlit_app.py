# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set the title of the app
st.title("💊 Pathways Prediction Platform")
st.write("Upload your data and explore data types and non-numeric values.")

# Initialize session state for selected model
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None

# Initialize session state for descriptions
if 'show_lr_desc' not in st.session_state:
    st.session_state['show_lr_desc'] = False
if 'show_rf_desc' not in st.session_state:
    st.session_state['show_rf_desc'] = False
if 'show_wsm_desc' not in st.session_state:
    st.session_state['show_wsm_desc'] = False

# Step 1: Upload CSV and download template
st.subheader("Step 1: Upload Your CSV File")

# Provide the download button for the CSV template
st.download_button(
    label="Need a template? Download the CSV Here 📄",
    data=pd.DataFrame({
        'acct_numb': ['123', '456', '789'],
        'acct_name': ['Account A', 'Account B', 'Account C'],
        'ProdA_sales_first12': [10000, 15000, 12000],
        'ProdA_units_first12': [100, 150, 120],
        'competition_sales_first12': [5000, 6000, 5500],
        'competition_units_first12': [50, 60, 55],
        'ProdA_sales_2022': [20000, 25000, 22000],
        'ProdA_units_2022': [200, 250, 220],
        'competition_sales_2022': [10000, 11000, 10500],
        'competition_units_2022': [100, 110, 105],
        'ProdA_sales_2023': [30000, 35000, 32000],
        'Total 2022 and 2023': [50000, 60000, 54000],
        'ProdA_units_2023': [300, 350, 320],
        'competition_sales_2023': [15000, 16000, 15500],
        'competition_units_2023': [150, 160, 155],
        'analog_1_adopt': ['low', 'medium', 'high'],
        'analog_2_adopt': ['medium', 'low', 'high'],
        'analog_3_adopt': ['high', 'medium', 'low'],
        'quintile_ProdA_totalsales': [1, 2, 1],
        'quintile_ProdB_opportunity': [3, 4, 5],
        'ability_to_influence': [0.7, 0.8, 0.75],
        'percentage_340B_adoption': [0.2, 0.3, 0.25]
    }).to_csv(index=False),
    file_name='csv_template.csv',
    mime='text/csv'
)

# Add a little space for better UI design
st.markdown("<br>", unsafe_allow_html=True)

# Upload box for CSV
uploaded_file = st.file_uploader("Now, choose your CSV file:", type="csv", label_visibility="visible")

# Default feature weights (for potential use later)
feature_weights = {
    'ProdA_sales_first12': 1.2,
    'ProdA_units_first12': 1.1,
    'competition_sales_first12': 0.9,
    'competition_units_first12': 0.8,
    'ProdA_sales_2022': 1.3,
    'ProdA_units_2022': 1.1,
    'competition_sales_2022': 0.9,
    'competition_units_2022': 0.8,
    'Total 2022 and 2023': 1.0,
    'ProdA_units_2023': 1.1,
    'competition_sales_2023': 0.9,
    'competition_units_2023': 0.8,
    'analog_1_adopt': 0.7,
    'analog_2_adopt': 0.6,
    'analog_3_adopt': 0.5,
    'quintile_ProdA_totalsales': 1.0,
    'quintile_ProdB_opportunity': 1.0,
    'ability_to_influence': 0.8,
    'percentage_340B_adoption': 0.6
}

# List of numeric and categorical columns
numeric_columns = [
    'ProdA_sales_first12', 'ProdA_units_first12', 'competition_sales_first12', 'competition_units_first12',
    'ProdA_sales_2022', 'ProdA_units_2022', 'competition_sales_2022', 'competition_units_2022',
    'ProdA_sales_2023', 'Total 2022 and 2023', 'ProdA_units_2023', 'competition_sales_2023',
    'competition_units_2023', 'percentage_340B_adoption'
]

categorical_columns = ['analog_1_adopt', 'analog_2_adopt', 'analog_3_adopt']

# Step 2: Model Selection (only if the file has been uploaded)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display the raw data
    st.dataframe(df.head())
    
    st.subheader("Step 2: Choose a Model")
    # Model selection buttons with descriptions as clickable links
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Linear Regression", key='lr'):
            st.session_state['selected_model'] = 'linear_regression'
        if st.button("Description", key='lr_desc'):
            st.session_state['show_lr_desc'] = not st.session_state['show_lr_desc']
        if st.session_state['show_lr_desc']:
            st.info("Linear Regression: Choose this model if you're working with between 10-50 lines of data.")

    with col2:
        if st.button("Run Random Forest", key='rf'):
            st.session_state['selected_model'] = 'random_forest'
        if st.button("Description", key='rf_desc'):
            st.session_state['show_rf_desc'] = not st.session_state['show_rf_desc']
        if st.session_state['show_rf_desc']:
            st.info("Random Forest: Choose this model if you're working with >50 lines of data and want to leverage predictive power.")

    with col3:
        if st.button("Run Weighted Scoring Model", key='wsm'):
            st.session_state['selected_model'] = 'weighted_scoring_model'
        if st.button("Description", key='wsm_desc'):
            st.session_state['show_wsm_desc'] = not st.session_state['show_wsm_desc']
        if st.session_state['show_wsm_desc']:
            st.info("Weighted Scoring Model: Choose this model if you're looking for analysis, not prediction.")

    # Execute selected model
    if st.session_state['selected_model'] == 'linear_regression':
        st.info("Running Linear Regression...")
        run_linear_regression(df, numeric_columns)
    elif st.session_state['selected_model'] == 'random_forest':
        st.info("Running Random Forest...")
        run_random_forest(df, numeric_columns)
    elif st.session_state['selected_model'] == 'weighted_scoring_model':
        st.info("Running Weighted Scoring Model...")
        run_weighted_scoring_model(df)

    # Add a "Run a New Model" button after the model has run
    if st.session_state['selected_model'] is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Run a New Model 🔄"):
            st.session_state['selected_model'] = None  # Reset the session state
            st.experimental_set_query_params()  # Clear content from previous run

