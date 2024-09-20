# Import necessary libraries
import streamlit as st
import pandas as pd

# Set the title of the app
st.title("⚕️ Pathways Prediction Platform 💊")
st.write("Upload your data and explore data types and non-numeric values.")

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
    non_numeric_data = {}
    for col in numeric_columns:
        # Remove dollar signs, commas, and percent symbols
        df[col] = df[col].replace({'\$': '', ',': '', '%': ''}, regex=True)
        
        # Convert to numeric (force invalid values to NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Log non-numeric data in numeric columns
        non_numeric_rows = df[df[col].isna()][col]
        if len(non_numeric_rows) > 0:
            non_numeric_data[col] = non_numeric_rows

    # If there are non-numeric values, log them to the user
    if non_numeric_data:
        st.warning("The following non-numeric values were found and will be removed:")
        for col, rows in non_numeric_data.items():
            st.write(f"In column '{col}':")
            st.write(rows)
    
    # Drop rows with NaN values in any of the numeric columns
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
