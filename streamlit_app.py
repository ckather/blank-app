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
