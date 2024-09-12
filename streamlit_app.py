import streamlit as st
import pandas as pd

# Set the title of the app
st.title("⚕️ Pathways 💊")
st.write("Let's use this to predict some drug adoption. 💊🤓")

# Load the CSV from the GitHub repository (path based on where you put it)
# For example, if it's in the 'data' folder:
default_csv_path = "data/sample_file.csv"

try:
    df = pd.read_csv(default_csv_path)
    st.write("Here is the default CSV data from the repository:")
    st.dataframe(df)
except FileNotFoundError:
    st.error("Default CSV file not found. Please upload your own CSV.")

# Create an upload button for custom CSV files (if the user wants to upload their own)
uploaded_file = st.file_uploader("Or, upload your CSV file")

# If a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here is the data from the uploaded file:")
    st.dataframe(df)


