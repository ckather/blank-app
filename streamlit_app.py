import streamlit as st

st.title("⚕️ Pathways 💊")
st.write(
    "Let's use this to predict some drug adoption. 💊🤓"
)
# Import necessary libraries
import streamlit as st
import pandas as pd

# Create an upload button
uploaded_file = st.file_uploader("Step 1. Choose your CSV file. See the bottom of this page for document requirements.")

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the file into a DataFrame (assuming it's a CSV file)
    df = pd.read_csv(uploaded_file)
    
    # Display the content of the uploaded file
    st.write("Here is the data from the uploaded file:")
    st.dataframe(df)
else:
    st.write("Please upload a CSV file.")

# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Streamlit app
st.title("Weighted Linear Regression App")

# Create an upload button for CSV file
uploaded_file = st.file_uploader("Upload your CSV file")

# Default feature weights
feature_weights = {
    'units 2023': 1.5,  # Example weight for 'units 2023'
    'adoption likelihood metric 1': 0.8,  # Example weight for 'adoption likelihood metric 1'
    'adoption likelihood metric 2': 1.0   # Example weight for 'adoption likelihood metric 2'
}

# If a file is uploaded
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
