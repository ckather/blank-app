# Import necessary libraries
import streamlit as st
import pandas as pd

# Set the title of the app
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🔍 CSV Debugging Tool</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Let's isolate the issue by inspecting the raw data.</h4>", unsafe_allow_html=True)

# Function to clean and convert 'high', 'medium', 'low' to 3, 2, 1
def convert_high_medium_low(df, columns):
    mapping = {'high': 3, 'medium': 2, 'low': 1}
    
    # Log to track problematic data
    problematic_rows = {}

    for col in columns:
        # Ensure column is in string format, force lowercase, and strip extra spaces
        df[col] = df[col].astype(str).str.lower().str.strip()

        # Output the unique values in the column to inspect
        st.write(f"Unique values in column '{col}' before mapping:", df[col].unique())

        # Map the values; if not in the mapping, mark them as NaN
        df[col] = df[col].map(mapping)

        # Check if any rows have problematic values (i.e., NaN after mapping)
        if df[col].isna().any():
            problematic_rows[col] = df[df[col].isna()][col]
    
    # If there are problematic rows, log them and output for the user to fix
    if problematic_rows:
        st.error(f"Conversion failed for some values. Please review the following rows:")
        for col, rows in problematic_rows.items():
            st.write(f"Problematic values in column '{col}':")
            st.write(rows)
    
    return df

# Step 1: Upload CSV
st.subheader("Step 1: Upload Your CSV File")
uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

# If a file is uploaded, proceed to process it
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Original Data")
    st.dataframe(df.head())  # Preview the original data
    
    # Specify the columns that contain "high", "medium", "low"
    categorical_columns = ['analog_1_adopt', 'analog_2_adopt', 'analog_3_adopt']
    
    st.subheader("Step 2: Convert 'low', 'medium', 'high' to Numeric")
    
    # Try to convert the categorical columns
    df = convert_high_medium_low(df, categorical_columns)
    
    # Display the modified DataFrame to see the results of the conversion
    st.subheader("Data After Conversion")
    st.dataframe(df.head())

    st.write("Full list of values in 'analog_1_adopt':", df['analog_1_adopt'].unique())
    st.write("Full list of values in 'analog_2_adopt':", df['analog_2_adopt'].unique())
    st.write("Full list of values in 'analog_3_adopt':", df['analog_3_adopt'].unique())

