# Import necessary libraries
import streamlit as st
import pandas as pd

# Set the title of the app
st.title("💊 Pathways Prediction Platform")
st.write("Upload your data and explore data types and non-numeric values.")

# Initialize session state for selected model and descriptions
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None

if 'show_description' not in st.session_state:
    st.session_state['show_description'] = {
        'linear_regression': False,
        'random_forest': False,
        'weighted_scoring_model': False
    }
# Step 1: Upload CSV and download template
st.subheader("Step 1: Upload Your CSV File")

# Provide the download button for the CSV template
st.download_button(
    label="Need a template? Download the CSV Here 📄",
    data=pd.DataFrame().to_csv(index=False),  # Empty DataFrame as placeholder
    file_name='csv_template.csv',
    mime='text/csv'
)

# Add a little space for better UI design
st.markdown("<br>", unsafe_allow_html=True)

# Upload box for CSV
uploaded_file = st.file_uploader(
    "Now, choose your CSV file:", type="csv", label_visibility="visible"
)
    # Define a function to create a styled hyperlink-like button
    def hyperlink_button(text, key):
        return st.button(
            text,
            key=key,
            help=None,
            args=None,
            kwargs=None
        )

    # Function to style buttons to look like hyperlinks
    def style_hyperlink_button(key):
        st.markdown(f"""
            <style>
            button[data-testid="stButton"][key="{key}"] {{
                background-color: transparent;
                color: orange;
                text-decoration: underline;
                border: none;
                padding: 0;
                margin-top: -10px;
            }}
            </style>
            """, unsafe_allow_html=True)

    with col1:
        if st.button("Run Linear Regression", key='lr'):
            st.session_state['selected_model'] = 'linear_regression'

        if hyperlink_button("Description", key='lr_desc'):
            st.session_state['show_description']['linear_regression'] = not st.session_state['show_description']['linear_regression']

        style_hyperlink_button('lr_desc')

        if st.session_state['show_description']['linear_regression']:
            st.info("Linear Regression: Choose this model if you're working with between 10-50 lines of data.")

    # Repeat for other models
    with col2:
        if st.button("Run Random Forest", key='rf'):
            st.session_state['selected_model'] = 'random_forest'

        if hyperlink_button("Description", key='rf_desc'):
            st.session_state['show_description']['random_forest'] = not st.session_state['show_description']['random_forest']

        style_hyperlink_button('rf_desc')

        if st.session_state['show_description']['random_forest']:
            st.info("Random Forest: Choose this model if you're working with >50 lines of data and want to leverage predictive power.")

    with col3:
        if st.button("Run Weighted Scoring Model", key='wsm'):
            st.session_state['selected_model'] = 'weighted_scoring_model'

        if hyperlink_button("Description", key='wsm_desc'):
            st.session_state['show_description']['weighted_scoring_model'] = not st.session_state['show_description']['weighted_scoring_model']

        style_hyperlink_button('wsm_desc')

        if st.session_state['show_description']['weighted_scoring_model']:
            st.info("Weighted Scoring Model: Choose this model if you're looking for analysis, not prediction.")
