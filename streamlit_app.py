# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set the title of the app
st.title("⚕️ Pathways Prediction Platform 💊")
st.write("Predict drug adoption using advanced machine learning models")

# Function to convert 'high', 'medium', 'low' to numeric (3, 2, 1)
def convert_categorical_columns(df, columns):
    mapping = {'high': 3, 'medium': 2, 'low': 1}
    
    for col in columns:
        # Force all values to lowercase strings and remove extra spaces
        df[col] = df[col].str.lower().str.strip()
        
        # Convert 'high', 'medium', 'low' to 3, 2, 1
        df[col] = df[col].map(mapping)
        
        # Check if conversion was successful, if not, drop rows with invalid data
        if df[col].isna().any():
            st.warning(f"Warning: Column '{col}' contains invalid values, those rows will be removed.")
            df = df.dropna(subset=[col])
    
    return df

# Function to clean numeric columns and remove unwanted symbols like $, %, and ,
def clean_numeric_columns(df, numeric_columns):
    for col in numeric_columns:
        # Remove dollar signs, commas, and percent symbols
        df[col] = df[col].replace({'\$': '', ',': '', '%': ''}, regex=True)
        
        # Convert to numeric (force invalid values to NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values in any of the numeric columns
    df = df.dropna(subset=numeric_columns)
    
    return df

# Function to run weighted linear regression
def run_weighted_linear_regression(df, feature_weights, numeric_columns, categorical_columns):
    # First, convert the categorical columns ('high', 'medium', 'low') to numeric
    df = convert_categorical_columns(df, categorical_columns)
    
    # Clean the numeric columns (remove unwanted symbols like $, %)
    df = clean_numeric_columns(df, numeric_columns)
    
    # Remove non-numeric columns like account numbers and names
    df = df.drop(columns=['acct_numb', 'acct_name'])
    
    # Make sure all columns are numeric
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Check for any remaining invalid rows
    if df[numeric_columns].isnull().any().any():
        st.error("Error: Some numeric columns still contain invalid data. These rows will be dropped.")
        df = df.dropna(subset=numeric_columns)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']

    # Apply feature weights to the numeric columns
    for feature in feature_weights:
        if feature in X.columns:
            X[feature] *= feature_weights[feature]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    # Output results
    results = {
        'Feature': list(X.columns) + ['Intercept', 'RMSE'],
        'Coefficient': list(lr.coef_) + [lr.intercept_, rmse]
    }
    results_df = pd.DataFrame(results)

    # Display results
    st.success("Linear Regression Results:")
    st.table(results_df)

# Function to run a Random Forest model
def run_random_forest(df, feature_weights, numeric_columns, categorical_columns):
    # Convert categorical columns ('high', 'medium', 'low') to numeric
    df = convert_categorical_columns(df, categorical_columns)
    
    # Clean numeric columns (remove $, %, etc.)
    df = clean_numeric_columns(df, numeric_columns)
    
    # Remove non-numeric columns like account numbers and names
    df = df.drop(columns=['acct_numb', 'acct_name'])

    # Make sure all columns are numeric
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Check for any remaining invalid rows
    if df[numeric_columns].isnull().any().any():
        st.error("Error: Some numeric columns still contain invalid data. These rows will be dropped.")
        df = df.dropna(subset=numeric_columns)
    
    # Prepare features and target
    X = df.drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']

    # Apply weights to the numeric columns
    for feature in feature_weights:
        if feature in X.columns:
            X[feature] *= feature_weights[feature]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions and calculate RMSE
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = mse_rf ** 0.5

    # Output feature importances and RMSE
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Display results
    st.success(f"Random Forest RMSE: {rmse_rf:.2f}")
    st.write("Feature Importances:")
    st.table(feature_importances)

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

# Default feature weights
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

# Process file upload and run regression
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())  # Preview the raw data
    
    st.subheader("Step 2: Run Your Weighted Linear Regression")
    if st.button('Run Linear Regression'):
        st.info("Running the regression model, please wait...")
        run_weighted_linear_regression(df, feature_weights, numeric_columns, categorical_columns)
        
        # Button to run Random Forest after Linear Regression completes
        st.subheader("Step 4: Run Your Random Forest Model")
        if st.button('Run Random Forest Model'):
            st.info("Running the Random Forest model, please wait...")
            run_random_forest(df, feature_weights, numeric_columns, categorical_columns)
