# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Define a function to wrap content with a thin border
def bordered_content(title, content):
    st.markdown(f"""
    <div style="border: 1px solid #D3D3D3; padding: 20px; border-radius: 10px;">
        <h4 style="color: #2E86C1; text-align: center;">{title}</h4>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Set the sleek title of the app with a professional design
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>⚕️ Pathways Prediction Platform 💊</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict drug adoption using advanced machine learning models</h4>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 3px solid #2E86C1;'>", unsafe_allow_html=True)

# Function to clean and ensure data validity
def clean_data(df):
    try:
        # Remove commas, dollar signs, and percentage symbols from relevant columns
        df = df.replace({'\$': '', ',': '', '%': ''}, regex=True)
        
        # Convert all columns to numeric where possible, with errors as NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values after processing
        df = df.dropna()
        
        if df.empty:
            raise ValueError("No valid data after cleaning.")
        
        return df
    
    except Exception as e:
        st.error(f"Error: {str(e)}. Please check your CSV file and ensure all columns contain numeric values.")
        return None

# Function to run weighted linear regression
def run_weighted_linear_regression(df, feature_weights):
    # Ensure columns required for regression exist
    required_columns = ['acct_numb', 'acct_name', 'ProdA_sales_2023']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing one or more required columns: {', '.join(required_columns)}")
        return

    # Clean and validate data
    df = clean_data(df)
    if df is None:
        return
    
    # Drop non-numeric columns like acct_numb and acct_name
    df = df.drop(columns=['acct_numb', 'acct_name'])

    st.info("Summary of the data used for regression:")
    st.dataframe(df.describe())

    st.info("Correlation Matrix between variables:")
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
    # Ensure columns required for Random Forest exist
    required_columns = ['ProdA_sales_2023']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing one or more required columns: {', '.join(required_columns)}")
        return

    st.subheader("Step 4: Running a Random Forest Machine Learning Model")

    # Clean and validate data
    df = clean_data(df)
    if df is None:
        return

    # Prepare features and target
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

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions and performance evaluation
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = mse_rf ** 0.5

    st.success(f"Random Forest Model RMSE: {rmse_rf:.2f}")
    st.write("Feature Importances:")
    
    # Feature Importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.table(feature_importances)

    # Provide download options for Random Forest results
    st.subheader("Download Random Forest Results")
    rf_importance_csv = feature_importances.to_csv(index=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Random Forest Feature Importances 🌲",
            data=rf_importance_csv,
            file_name='random_forest_importances.csv',
            mime='text/csv'
        )
    with col2:
        rf_predictions_csv = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf}).to_csv(index=False)
        st.download_button(
            label="Download Random Forest Predictions 🔍",
            data=rf_predictions_csv,
            file_name='random_forest_predictions.csv',
            mime='text/csv'
        )

# Step 1: Upload CSV
bordered_content("Step 1: Upload Your CSV File", """
    <div style='text-align: center;'>
        <p>Choose your CSV file below</p>
    </div>
""")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

# Provide template download button
csv_template = generate_csv_template()
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

# Process file upload and run regression
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    bordered_content("Uploaded Data Preview", df.head().to_html(index=False))
    
    st.subheader("Step 2: Run Your Weighted Linear Regression")
    if st.button('Run Linear Regression'):
        st.info("Running the regression model, please wait...")
        run_weighted_linear_regression(df, feature_weights)
        
        # Button to run Random Forest after Linear Regression completes
        st.subheader("Step 4: Run Your Random Forest Model")
        if st.button('Run Random Forest Model'):
            st.info("Running the Random Forest model, please wait...")
            run_random_forest(df, feature_weights)
