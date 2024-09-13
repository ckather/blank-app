# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
    df = df.drop(columns=['acct_numb', 'acct_name'])
    df = df.replace({"high": 3, "medium": 2, "low": 1})
    
    sales_columns = ['ProdA_sales_first12', 'competition_sales_first12', 'ProdA_sales_2022', 'competition_sales_2022', 'ProdA_sales_2023', 'Total 2022 and 2023', 'competition_sales_2023']
    for col in sales_columns:
        df[col] = df[col].replace({'\$': '', ',': ''}, regex=True)
    df['percentage_340B_adoption'] = df['percentage_340B_adoption'].str.replace('%', '').astype(float)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    if df.empty:
        st.error("Error: No valid data after processing. Please check your CSV file and ensure all columns contain numeric values.")
        return

    st.info("Here is a summary of the data used for regression:")
    st.dataframe(df.describe())

    st.info("Correlation Matrix to identify relationships between variables:")
    correlation_matrix = df.corr()
    st.dataframe(correlation_matrix)

    X = df.drop(columns=['ProdA_sales_2023'])
    y = df['ProdA_sales_2023']

    for feature in feature_weights:
        if feature in X.columns:
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
        'Feature': list(df.drop(columns=['ProdA_sales_2023']).columns) + ['Intercept', 'RMSE'],
        'Coefficient': list(lr.coef_) + [lr.intercept_, rmse]
    }
    results_df = pd.DataFrame(results)

    st.success("Linear Regression Results:")
    st.table(results_df)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
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

# Step 1: Upload CSV
st.subheader("Step 1: Upload Your CSV File")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.info("Here is a preview of your uploaded data:")
    st.dataframe(df.head())

    st.subheader("Step 2: Run Your Weighted Linear Regression")
    if st.button('Run Linear Regression'):
        st.info("Running the regression model, please wait...")
        run_weighted_linear_regression(df, feature_weights)
