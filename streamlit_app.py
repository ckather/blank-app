# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set the title of the app
st.title("⚕️ Pathways Prediction Platform 💊")
st.write("Upload your data and explore data types and non-numeric values.")

# Step 1: Upload CSV and download template
st.subheader("Step 1: Upload Your CSV File")

# Provide the download button for the CSV template with the new label
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

# Upload box that spans full width
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

# Function to run linear regression
def run_linear_regression(df, numeric_columns):
    X = df[numeric_columns].drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    st.success(f"Linear Regression RMSE: {rmse:.2f}")

# Function to run random forest
def run_random_forest(df, numeric_columns):
    X = df[numeric_columns].drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = mse_rf ** 0.5
    st.success(f"Random Forest RMSE: {rmse_rf:.2f}")

# Process file upload and log issues
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display the raw data
    st.dataframe(df.head())
    
    # Convert categorical columns (high, medium, low) to numeric
    def convert_categorical_columns(df, columns):
        mapping = {'high': 3, 'medium': 2, 'low': 1}
        for col in columns:
            df[col] = df[col].str.lower().str.strip().map(mapping)
            if df[col].isna().any():
                st.warning(f"Column '{col}' contains invalid values and those rows will be dropped.")
                df = df.dropna(subset=[col])
        return df
    
    df = convert_categorical_columns(df, categorical_columns)
    
    # Clean numeric columns (removing $, %, etc.)
    def clean_numeric_columns(df, numeric_columns):
        non_numeric_data = {}
        for col in numeric_columns:
            df[col] = df[col].replace({'\$': '', ',': '', '%': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            non_numeric_rows = df[df[col].isna()][col]
            if len(non_numeric_rows) > 0:
                non_numeric_data[col] = non_numeric_rows
        if non_numeric_data:
            st.warning("Non-numeric values were found and will be removed:")
            for col, rows in non_numeric_data.items():
                st.write(f"In column '{col}':")
                st.write(rows)
        df = df.dropna(subset=numeric_columns)
        return df
    
    df = clean_numeric_columns(df, numeric_columns)
    
    # Display the cleaned data
    st.write("Cleaned data:")
    st.dataframe(df.head())
    
    # Step: Ask user which model to run
    st.subheader("Step 2: Choose a Model")
    
    # Track user selection
    selected_model = st.session_state.get('selected_model', None)
    
    if selected_model is None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Run Linear Regression'):
                st.session_state['selected_model'] = 'linear_regression'
                run_linear_regression(df, numeric_columns)
        with col2:
            if st.button('Run Random Forest'):
                st.session_state['selected_model'] = 'random_forest'
                run_random_forest(df, numeric_columns)
    elif st.session_state['selected_model'] == 'linear_regression':
        st.info("You have chosen to run the Linear Regression model.")
        run_linear_regression(df, numeric_columns)
    elif st.session_state['selected_model'] == 'random_forest':
        st.info("You have chosen to run the Random Forest model.")
        run_random_forest(df, numeric_columns)
