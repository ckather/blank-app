# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set the title of the app
st.title("⚕️ Pathways 💊")
st.write("Let's use this to predict some drug adoption. 💊🤓")

# Function to generate CSV template with disclaimer
def generate_csv_template():
    # Create a disclaimer row
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

    # Create the data with placeholder values
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

    # Convert template data to a DataFrame
    template_df = pd.DataFrame(template_data)

    # Combine the disclaimer with the actual data
    combined_df = pd.concat([disclaimer, template_df], ignore_index=True)
    
    # Return the CSV as a string
    return combined_df.to_csv(index=False)

# Function to run weighted linear regression
def run_weighted_linear_regression(df, feature_weights):
    # Drop irrelevant columns (like acct_numb, acct_name)
    df = df.drop(columns=['acct_numb', 'acct_name'])

    # Replace "high", "medium", "low" with 3, 2, 1 respectively
    df = df.replace({"high": 3, "medium": 2, "low": 1})

    # Ensure all relevant columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Display the processed DataFrame for debugging
    st.write("Processed DataFrame (after converting strings and numeric conversion):")
    st.dataframe(df)

    # Drop rows with NaN values (optional, depending on how you want to handle missing data)
    df = df.dropna()

    # Check if the dataset is empty after processing
    if df.empty:
        st.error("Error: No valid data after processing. Please check your CSV file and ensure all columns contain numeric values.")
        return

    # Separate features (X) and target (y) - assuming target is 'ProdA_sales_2023'
    X = df.drop(columns=['ProdA_sales_2023'])  # Exclude target column
    y = df['ProdA_sales_2023']  # Target column

    # Apply weights to the features
    for feature in feature_weights:
        if feature in X.columns:
            X[feature] *= feature_weights[feature]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if there is enough data for training
    if X_train.empty or X_test.empty:
        st.error("Error: Not enough data to train the model. Please provide more valid rows.")
        return

    # Initialize the linear regression model
    lr = LinearRegression()

    # Train the model
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Calculate the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    # Output the coefficients, intercept, and RMSE
    st.write("Linear Regression Coefficients:", lr.coef_)
    st.write("Linear Regression Intercept:", lr.intercept_)
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# Streamlit app setup
st.subheader("Step 1. Upload your CSV file")
uploaded_file = st.file_uploader("Please upload a CSV file.")

# Provide the CSV template download button below the file uploader
csv_template = generate_csv_template()
st.download_button(
    label="Need the CSV template? Download it here! 📄",
    data=csv_template,
    file_name='csv_template.csv',
    mime='text/csv'
)

# Default feature weights for the weighted linear regression
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

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("Here is a preview of the uploaded data:")
    st.dataframe(df.head())
    
    # Add a step title for running the linear regression
    st.subheader("Step 2. Run your linear regression")
    
    # Add a "Run" button
    if st.button('Run Weighted Linear Regression'):
        st.write("Running weighted linear regression...")
        run_weighted_linear_regression(df, feature_weights)

