# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set up CSS styles for a sleeker look
st.markdown(
    """
    <style>
    .centered-logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        padding-bottom: 20px;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .css-1aumxhk {
        background-color: #f0f2f6; /* Custom background color */
    }

    .title-text {
        font-family: 'Arial', sans-serif;
        font-size: 48px;
        font-weight: bold;
        color: #004080; /* Custom title color */
        text-align: center;
    }

    .subheader {
        font-size: 24px;
        color: #004080;
        margin-bottom: 20px;
    }

    .stButton>button {
        background-color: #004080;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Display the logo using CSS for centering
st.markdown('<img src="path/to/your/logo.png" class="centered-logo">', unsafe_allow_html=True)

# Title with custom CSS styling
st.markdown('<div class="title-text">Pathways Prediction Platform</div>', unsafe_allow_html=True)

st.write("Upload your data and explore data types and non-numeric values.")

# Function to convert 'high', 'medium', 'low' to numeric (3, 2, 1)
def convert_categorical_columns(df, columns):
    mapping = {'high': 3, 'medium': 2, 'low': 1}
    
    for col in columns:
        df[col] = df[col].str.lower().str.strip()
        df[col] = df[col].map(mapping)
        if df[col].isna().any():
            st.warning(f"Warning: Column '{col}' contains invalid values, those rows will be removed.")
            df = df.dropna(subset=[col])
    
    return df

# Function to clean numeric columns and remove unwanted symbols
def clean_numeric_columns(df, numeric_columns):
    non_numeric_data = {}
    for col in numeric_columns:
        df[col] = df[col].replace({'\$': '', ',': '', '%': ''}, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        non_numeric_rows = df[df[col].isna()][col]
        if len(non_numeric_rows) > 0:
            non_numeric_data[col] = non_numeric_rows

    if non_numeric_data:
        st.warning("The following non-numeric values were found and will be removed:")
        for col, rows in non_numeric_data.items():
            st.write(f"In column '{col}':")
            st.write(rows)
    
    df = df.dropna(subset=numeric_columns)
    
    return df

# Step 1: Upload CSV
st.subheader("Step 1: Upload Your CSV File")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

# Provide template download button
csv_template = pd.DataFrame({
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
}).to_csv(index=False)

st.download_button(
    label="Download CSV Template 📄",
    data=csv_template,
    file_name='csv_template.csv',
    mime='text/csv'
)

# List of numeric and categorical columns
numeric_columns = [
    'ProdA_sales_first12', 'ProdA_units_first12', 'competition_sales_first12', 'competition_units_first12',
    'ProdA_sales_2022', 'ProdA_units_2022', 'competition_sales_2022', 'competition_units_2022',
    'ProdA_sales_2023', 'Total 2022 and 2023', 'ProdA_units_2023', 'competition_sales_2023',
    'competition_units_2023', 'percentage_340B_adoption'
]

categorical_columns = ['analog_1_adopt', 'analog_2_adopt', 'analog_3_adopt']

# Process file upload and log issues
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display the raw data
    st.dataframe(df.head())
    
    # Convert categorical columns (high, medium, low) to numeric
    df = convert_categorical_columns(df, categorical_columns)
    
    # Clean numeric columns (removing $, %, etc.)
    df = clean_numeric_columns(df, numeric_columns)
    
    # Display the cleaned data
    st.write("Cleaned data:")
    st.dataframe(df.head())

# Function to run weighted linear regression
def run_weighted_linear_regression(df, feature_weights, numeric_columns, categorical_columns):
    df = convert_categorical_columns(df, categorical_columns)
    df = clean_numeric_columns(df, numeric_columns)
    
    df = df.drop(columns=['acct_numb', 'acct_name'])

    X = df.drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']

    for feature in feature_weights:
        if feature in X.columns:
            X[feature] = pd.to_numeric(X[feature], errors='coerce')  
            X[feature] *= feature_weights[feature]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    results = {
        'Feature': list(X.columns) + ['Intercept', 'RMSE'],
        'Coefficient': list(lr.coef_) + [lr.intercept_, rmse]
    }
    results_df = pd.DataFrame(results)

    st.success("Linear Regression Results:")
    st.table(results_df)

# Function to run a Random Forest model
def run_random_forest(df, feature_weights, numeric_columns, categorical_columns):
    df = convert_categorical_columns(df, categorical_columns)
    df = clean_numeric_columns(df, numeric_columns)

    df = df.drop(columns=['acct_numb', 'acct_name'])

    X = df.drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']

    for feature in feature_weights:
        if feature in X.columns:
            X[feature] = pd.to_numeric(X[feature], errors='coerce')  
            X[feature] *= feature_weights[feature]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = mse_rf ** 0.5

    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.success(f"Random Forest RMSE: {rmse_rf:.2f}")
    st.write("Feature Importances:")
    st.table(feature_importances)

# Default feature weights for linear regression
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

# Step 2: Run regression models (optional depending on file upload)
st.subheader("Step 2: Run Your Weighted Linear Regression")
if uploaded_file is not None and st.button('Run Linear Regression'):
    run_weighted_linear_regression(df, feature_weights, numeric_columns, categorical_columns)

st.subheader("Step 4: Run Your Random Forest Model")
if uploaded_file is not None and st.button('Run Random Forest'):
    run_random_forest(df, feature_weights, numeric_columns, categorical_columns)
