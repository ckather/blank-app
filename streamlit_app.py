# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set the title of the app with a modern design
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>⚕️ Pathways 💊</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict drug adoption using weighted linear regression</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Function to generate CSV template with disclaimer
def generate_csv_template():
    disclaimer = pd.DataFrame({
        'acct_numb': ['Disclaimer: These are placeholder numbers. Please replace with your own data.'],
        'acct_name': [''],
        'ProdA_sales_first12': [''],
        'ProdA_units_first12': [''],
        'competition_sales_first12': [''],
        'competition_units_first12': [''],
        'ProdA_sales_2022': [''],
        'ProdA_units_2022': [''],
        'competition_sales_2022': [''],
        'competition_units_2022': [''],
        'ProdA_sales_2023': [''],
        'Total 2022 and 2023': [''],
        'ProdA_units_2023': [''],
        'competition_sales_2023': [''],
        'competition_units_2023': [''],
        'analog_1_adopt': [''],
        'analog_2_adopt': [''],
        'analog_3_adopt': [''],
        'quintile_ProdA_totalsales': [''],
        'quintile_ProdB_opportunity': [''],
        'ability_to_influence': [''],
        'percentage_340B_adoption': ['']
    })
    
    template_data = {
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
        'analog_1_adopt': [0.6, 0.7, 0.65],
        'analog_2_adopt': [0.5, 0.6, 0.55],
        'analog_3_adopt': [0.4, 0.5, 0.45],
        'quintile_ProdA_totalsales': [1, 2, 1],
        'quintile_ProdB_opportunity': [3, 4, 5],
        'ability_to_influence': [0.7, 0.8, 0.75],
        'percentage_340B_adoption': [0.2, 0.3, 0.25]
    }

    template_df = pd.DataFrame(template_data)
    combined_df = pd.concat([disclaimer, template_df], ignore_index=True)
    
    return combined_df.to_csv(index=False)

# Function to run weighted linear regression
def run_weighted_linear_regression(df, feature_weights):
    # Drop non-numeric columns like acct_numb and acct_name
    df = df.drop(columns=['acct_numb', 'acct_name'])
    
    # Replace "high", "medium", "low" with numeric values
    df = df.replace({"high": 3, "medium": 2, "low": 1})
    
    # Remove commas and dollar signs from sales columns
    sales_columns = ['ProdA_sales_first12', 'competition_sales_first12', 'ProdA_sales_2022', 'competition_sales_2022', 'ProdA_sales_2023', 'Total 2022 and 2023', 'competition_sales_2023']
    for col in sales_columns:
        df[col] = df[col].replace({'\$': '', ',': ''}, regex=True)
    
    # Remove percentage signs and convert to numeric
    df['percentage_340B_adoption'] = df['percentage_340B_adoption'].str.replace('%', '').astype(float)

    # Ensure all relevant columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()
    
    if df.empty:
        st.error("Error: No valid data after processing. Please check your CSV file and ensure all columns contain numeric values.")
        return

    st.info("Here is a summary of the data used for regression:")
    st.dataframe(df.describe())

    st.info("Correlation Matrix to identify relationships between variables:")
    correlation_matrix = df.corr()
    st.dataframe(correlation_matrix)

    # Separate features (X) and target (y)
    X = df.drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']

    # Apply weights to numeric columns only
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

    # Prepare results for display
    results = {
        'Feature': list(X.columns) + ['Intercept', 'RMSE'],
        'Coefficient': list(lr.coef_) + [lr.intercept_, rmse]
    }
    results_df = pd.DataFrame(results)

    st.success("Linear Regression Results:")
    st.table(results_df)

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
            file_name='correlation_matrix.csv',
            mime='text/csv'
        )
    with col2:
        st.download_button(
            label="Download Regression Results 📉",
            data=results_csv,
            file_name='regression_results.csv',
            mime='text/csv'
        )

# Function to run a Random Forest model
def run_random_forest(df, feature_weights):
    st.subheader("Step 4: Running a Random Forest Machine Learning Model")

    # Prepare features and target
    X = df.drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']

    for feature in feature_weights:
        if feature in X.columns:
            X[feature] *= feature_weights[feature]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions and performance evaluation
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = mse_rf ** 0.5

    st.success(f"Random Forest Model RMSE: {rmse_rf:.2f}")
    st.write("Feature Importances:")
    
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.table(feature_importances)

# Step 1: Upload CSV
st.subheader("Step 1: Upload Your CSV File")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

# Provide template download button
csv_template = generate_csv_template()
st.download_button(
    label="Need a CSV template? Download here 📄",
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

# Use session state to store user action
if 'analysis_decision' not in st.session_state:
    st.session_state.analysis_decision = None

# Process file upload and run regression
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.info("Here is a preview of your uploaded data:")
    st.dataframe(df.head())

    st.subheader("Step 2: Run Your Weighted Linear Regression")
    if st.button('Run Linear Regression'):
        st.info("Running the regression model, please wait...")
        run_weighted_linear_regression(df, feature_weights)
        
        # Ask the user if they want to proceed with further analysis
        st.subheader("Question: Do you want to proceed with further analysis?")
        
        # Yes and No buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, proceed 🔄"):
                st.session_state.analysis_decision = 'yes'
        with col2:
            if st.button("No, stop ❌"):
                st.session_state.analysis_decision = 'no'
                st.experimental_rerun()  # Refresh the page if "No" is pressed
        
        # Handle button clicks without resetting the page view
        if st.session_state.analysis_decision == 'yes':
            st.success("Proceeding to Step 4: Running a Random Forest Machine Learning model.")
            run_random_forest(df, feature_weights)
