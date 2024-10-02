# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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
    data=pd.DataFrame({
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
    }).to_csv(index=False),
    file_name='csv_template.csv',
    mime='text/csv'
)

# Add a little space for better UI design
st.markdown("<br>", unsafe_allow_html=True)

# Upload box for CSV
uploaded_file = st.file_uploader(
    "Now, choose your CSV file:", type="csv", label_visibility="visible"
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

# Define model functions

def run_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    st.subheader("Linear Regression Results")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    
    # Plot Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

def run_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    st.subheader("Random Forest Results")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    
    # Plot Feature Importance
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importances.plot(kind='bar', ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)
    
    # Plot Actual vs Predicted
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, predictions, alpha=0.7)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Actual vs Predicted")
    st.pyplot(fig2)

def run_weighted_scoring_model(df, feature_weights, target_column):
    st.subheader("Weighted Scoring Model Results")
    
    # Calculate weighted score
    score = 0
    for feature, weight in feature_weights.items():
        if feature in df.columns:
            score += df[feature] * weight
        else:
            st.warning(f"Feature '{feature}' not found in the data. Skipping its weight.")
    
    df['Weighted_Score'] = score
    
    # Correlation with target
    correlation = df['Weighted_Score'].corr(df[target_column])
    st.write(f"**Correlation between Weighted Score and {target_column}:** {correlation:.2f}")
    
    # Display top accounts based on score
    top_n = st.slider("Select number of top accounts to display", min_value=5, max_value=20, value=10, step=1)
    top_accounts = df[['acct_numb', 'acct_name', 'Weighted_Score', target_column]].sort_values(by='Weighted_Score', ascending=False).head(top_n)
    st.dataframe(top_accounts)
    
    # Plot Weighted Score vs Target
    fig, ax = plt.subplots()
    ax.scatter(df['Weighted_Score'], df[target_column], alpha=0.7)
    ax.set_xlabel("Weighted Score")
    ax.set_ylabel(target_column)
    ax.set_title("Weighted Score vs Actual")
    st.pyplot(fig)

# Step 2: Model Selection (only if the file has been uploaded)
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        # Let user select the target column
        possible_targets = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        default_target = 'Total 2022 and 2023' if 'Total 2022 and 2023' in possible_targets else possible_targets[0]
        target_column = st.selectbox("Select the Target Column for Prediction", options=possible_targets, index=possible_targets.index(default_target))
        
        # Identify target and features
        if target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found in the uploaded data.")
        else:
            # Separate features and target
            feature_columns = [col for col in df.columns if col not in [target_column, 'acct_numb', 'acct_name']]
            X = df[feature_columns]
            y = df[target_column]

            # Handle categorical variables using one-hot encoding
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

            # Handle missing values (simple strategy: fill with mean for numeric, mode for categorical)
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col].fillna(X[col].mode()[0], inplace=True)
                else:
                    X[col].fillna(X[col].mean(), inplace=True)

            # Split the data into training and testing sets
            test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            st.write(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

            st.subheader("Step 2: Choose a Model")

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
                    button[data-testid="stButton"][data-key="{key}"] {{
                        background-color: transparent;
                        color: orange;
                        text-decoration: underline;
                        border: none;
                        padding: 0;
                        margin-top: -10px;
                    }}
                    </style>
                    """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Run Linear Regression", key='lr'):
                    st.session_state['selected_model'] = 'linear_regression'

                if hyperlink_button("Description", key='lr_desc'):
                    st.session_state['show_description']['linear_regression'] = not st.session_state['show_description']['linear_regression']

                style_hyperlink_button('lr_desc')

                if st.session_state['show_description']['linear_regression']:
                    st.info("**Linear Regression:** Choose this model if you're working with between 10-50 lines of data.")

            with col2:
                if st.button("Run Random Forest", key='rf'):
                    st.session_state['selected_model'] = 'random_forest'

                if hyperlink_button("Description", key='rf_desc'):
                    st.session_state['show_description']['random_forest'] = not st.session_state['show_description']['random_forest']

                style_hyperlink_button('rf_desc')

                if st.session_state['show_description']['random_forest']:
                    st.info("**Random Forest:** Choose this model if you're working with >50 lines of data and want to leverage predictive power.")

            with col3:
                if st.button("Run Weighted Scoring Model", key='wsm'):
                    st.session_state['selected_model'] = 'weighted_scoring_model'

                if hyperlink_button("Description", key='wsm_desc'):
                    st.session_state['show_description']['weighted_scoring_model'] = not st.session_state['show_description']['weighted_scoring_model']

                style_hyperlink_button('wsm_desc')

                if st.session_state['show_description']['weighted_scoring_model']:
                    st.info("**Weighted Scoring Model:** Choose this model if you're looking for analysis, not prediction.")

            # Execute selected model with loading spinner
            if st.session_state['selected_model'] == 'linear_regression':
                with st.spinner("Training Linear Regression model..."):
                    run_linear_regression(X_train, X_test, y_train, y_test)
            elif st.session_state['selected_model'] == 'random_forest':
                with st.spinner("Training Random Forest model..."):
                    run_random_forest(X_train, X_test, y_train, y_test)
            elif st.session_state['selected_model'] == 'weighted_scoring_model':
                with st.spinner("Calculating Weighted Scoring Model..."):
                    run_weighted_scoring_model(df, feature_weights, target_column)

            # Add a "Run a New Model" button after the model has run
            if st.session_state['selected_model'] is not None:
                st.markdown("<hr>", unsafe_allow_html=True)
                if st.button("Run a New Model 🔄"):
                    # Reset the session state
                    st.session_state['selected_model'] = None
                    st.session_state['show_description'] = {
                        'linear_regression': False,
                        'random_forest': False,
                        'weighted_scoring_model': False
                    }
                    st.experimental_rerun()

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
