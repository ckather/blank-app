# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set the title of the app
st.title("⚕️ Pathways 💊")
st.write("Let's use this to predict some drug adoption. 💊🤓")

# Function to generate CSV template
def generate_csv_template():
    template_data = {
        'account number': ['123', '456', '789'],
        'sales 2023': [10000, 15000, 12000],
        'units 2023': [100, 150, 120],
        'adoption likelihood metric 1': [0.5, 0.7, 0.6],
        'adoption likelihood metric 2': [0.3, 0.4, 0.5]
    }
    template_df = pd.DataFrame(template_data)
    return template_df.to_csv(index=False)

# Function to run weighted linear regression
def run_weighted_linear_regression(df, feature_weights):
    # Drop 'account number' since it's not needed for the regression
    df = df.drop(columns=['account number'])
    
    # Separate features (X) and target (y)
    X = df[['units 2023', 'adoption likelihood metric 1', 'adoption likelihood metric 2']]
    y = df['sales 2023']
    
    # Apply weights to the features
    X_weighted = X.copy()
    X_weighted['units 2023'] *= feature_weights['units 2023']
    X_weighted['adoption likelihood metric 1'] *= feature_weights['adoption likelihood metric 1']
    X_weighted['adoption likelihood metric 2'] *= feature_weights['adoption likelihood metric 2']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=42)
    
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

# Step 1: Upload CSV File
st.subheader("Step 1. Upload your CSV file")
uploaded_file = st.file_uploader("Please upload a CSV file.")

# Step 1b: Download the CSV template (this appears **below** the file uploader)
csv_template = generate_csv_template()
st.download_button(
    label="Need the CSV template? Download it here! 🗂️",
    data=csv_template,
    file_name='csv_template.csv',
    mime='text/csv'
)

# Default feature weights for the weighted linear regression
feature_weights = {
    'units 2023': 1.5,  # Example weight for 'units 2023'
    'adoption likelihood metric 1': 0.8,  # Example weight for 'adoption likelihood metric 1'
    'adoption likelihood metric 2': 1.0   # Example weight for 'adoption likelihood metric 2'
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
else:
    st.write("Please upload a CSV file.")
