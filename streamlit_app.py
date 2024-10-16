# app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set the page configuration
st.set_page_config(page_title="💊 Behavior Prediction Platform 💊", layout="wide")

# Initialize session state variables for navigation and selections
if 'step' not in st.session_state:
    st.session_state.step = 1  # Current step: 1 to 5

if 'df' not in st.session_state:
    st.session_state.df = None  # Uploaded DataFrame

if 'target_column' not in st.session_state:
    st.session_state.target_column = None  # User-selected target variable

if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []  # Selected independent variables

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None  # Selected model

if 'normalized_weights' not in st.session_state:
    st.session_state.normalized_weights = None  # Normalized weights

if 'dependent_variable_needed' not in st.session_state:
    st.session_state.dependent_variable_needed = False  # Flag for dependent variable selection

# Function to reset the app to Step 1
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 1

# Function to reset to Step 3 to allow running another model
def reset_to_step_3():
    st.session_state.step = 3

# Function to advance to the next step
def next_step():
    if st.session_state.step == 1 and st.session_state.df is None:
        st.warning("⚠️ Please upload a CSV file before proceeding.")
    elif st.session_state.step == 3:
        if st.session_state.selected_model is None:
            st.warning("⚠️ Please select a model before proceeding.")
        else:
            if st.session_state.selected_model == 'linear_regression':
                st.session_state.dependent_variable_needed = True
                st.session_state.step = 4
            else:
                # Preprocess data and proceed to results
                preprocess_data()
                st.session_state.step = 5
    elif st.session_state.step == 4:
        if st.session_state.dependent_variable_needed:
            if 'target_column' not in st.session_state or st.session_state.target_column is None:
                st.warning("⚠️ Please select a dependent variable before proceeding.")
            else:
                preprocess_data_with_target()
                st.session_state.step = 5
        else:
            st.session_state.step = 5
    elif st.session_state.step < 5:
        st.session_state.step += 1

# Function to go back to the previous step
def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1

# Define functions to set the selected model
def select_linear_regression():
    st.session_state.selected_model = 'linear_regression'

def select_weighted_scoring():
    st.session_state.selected_model = 'weighted_scoring_model'

def select_random_forest():
    st.session_state.selected_model = 'random_forest'

# Define mappings for categorical features
categorical_mappings = {
    'analog_1_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_2_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_3_adopt': {'low': 1, 'medium': 2, 'high': 3}
}

# Define helper functions

def encode_categorical_features(df, mappings):
    """
    Encodes categorical features based on provided mappings.
    """
    for feature, mapping in mappings.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping)
            if df[feature].isnull().any():
                st.warning(f"Some values in '{feature}' couldn't be mapped and are set to NaN.")
                df[feature].fillna(df[feature].mode()[0], inplace=True)
    return df

def generate_account_adoption_rank(df):
    """
    Generates the 'Account Adoption Rank Order' based on total sales across different periods.
    If 'Total_2022_and_2023' is missing, it calculates it using available sales columns.
    """
    # Define the columns to sum for ranking
    sales_columns = [
        'ProdA_sales_first12',
        'ProdA_sales_2022',
        'ProdA_sales_2023',
        'competition_sales_first12',
        'competition_sales_2022',
        'competition_sales_2023'
    ]

    # Check if 'Total_2022_and_2023' exists; if not, compute it
    if 'Total_2022_and_2023' not in df.columns:
        required_for_total = ['ProdA_sales_2022', 'ProdA_sales_2023', 'competition_sales_2022', 'competition_sales_2023']
        missing_total_cols = [col for col in required_for_total if col not in df.columns]
        if missing_total_cols:
            st.error(f"❌ To compute 'Total_2022_and_2023', the following columns are missing: {', '.join(missing_total_cols)}")
            st.stop()
        df['Total_2022_and_2023'] = df['ProdA_sales_2022'] + df['ProdA_sales_2023'] + df['competition_sales_2022'] + df['competition_sales_2023']
        st.info("'Total_2022_and_2023' column was missing and has been computed automatically.")
    else:
        st.info("'Total_2022_and_2023' column found in the uploaded file.")

    # Add 'Total_2022_and_2023' to sales_columns if not already present
    if 'Total_2022_and_2023' not in sales_columns:
        sales_columns.append('Total_2022_and_2023')

    # Check if all required sales columns are present
    missing_columns = [col for col in sales_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ The following required columns are missing to generate 'Account Adoption Rank Order': {', '.join(missing_columns)}")
        st.stop()

    # Calculate total sales
    df['Total_Sales'] = df[sales_columns].sum(axis=1)

    # Generate rank order (1 being highest sales)
    df['Account Adoption Rank Order'] = df['Total_Sales'].rank(method='dense', ascending=False).astype(int)

    return df

def preprocess_data():
    df = st.session_state.df
    selected_features = st.session_state.selected_features
    X = df[selected_features].copy()

    # Handle categorical variables using one-hot encoding
    selected_categorical = [col for col in selected_features if df[col].dtype == 'object']
    if selected_categorical:
        X = pd.get_dummies(X, columns=selected_categorical, drop_first=True)

    # Handle missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].mean(), inplace=True)

    # Handle infinite values
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    X = X.dropna()
    # Store preprocessed data in session state for use in Step 5
    st.session_state.X = X

def preprocess_data_with_target():
    df = st.session_state.df
    target_column = st.session_state.target_column
    X = st.session_state.X
    y = df[target_column]

    y = pd.to_numeric(y, errors='coerce')
    y = y.loc[X.index]  # Align y with X after preprocessing
    st.session_state.y = y

def run_selected_model():
    """
    Executes the selected model based on user input and advances to the next step.
    """
    selected_model = st.session_state.selected_model

    if selected_model == 'linear_regression':
        st.session_state.dependent_variable_needed = True
        st.session_state.step = 4  # Proceed to dependent variable selection
    else:
        st.session_state.dependent_variable_needed = False
        # Prepare data and proceed to results
        preprocess_data()
        st.session_state.step = 5  # Proceed to results

def render_sidebar():
    """
    Renders the instructions sidebar with step highlighting.
    """
    base_steps = [
        "Upload CSV File",
        "Select Independent Variables",
        "Choose Model & Assign Weights",
        "Your Results"
    ]

    if st.session_state.selected_model == 'linear_regression':
        step_titles = base_steps[:3] + ["Choose Dependent Variable", "Your Results"]
    else:
        step_titles = base_steps

    current_step = st.session_state.step

    st.sidebar.title("📖 Navigation")

    for i, title in enumerate(step_titles, 1):
        if i == current_step:
            st.sidebar.markdown(f"### ✅ **Step {i}: {title}**")
        elif i < current_step:
            st.sidebar.markdown(f"### ✅ Step {i}: {title}")
        else:
            st.sidebar.markdown(f"### Step {i}: {title}")

# Render the sidebar with step highlighting
render_sidebar()

# Main app logic based on current step

# Step 1: Upload CSV and Download Template
if st.session_state.step == 1:
    st.title("💊 Behavior Prediction Platform 💊")
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
            # 'Total_2022_and_2023' is intentionally omitted
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

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose your CSV file:", type="csv", label_visibility="visible"
    )

    # Process the uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df  # Store in session state

            # Check and generate 'Account Adoption Rank Order'
            df = generate_account_adoption_rank(df)
            st.session_state.df = df  # Update in session state

            st.success("✅ File uploaded and 'Account Adoption Rank Order' generated successfully!")
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

# Step 2: Select Independent Variables
elif st.session_state.step == 2:
    st.title("💊 Behavior Prediction Platform 💊")
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No data found. Please go back to Step 1 and upload your CSV file.")
    else:
        df = st.session_state.df
        st.subheader("Step 2: Select Independent Variables")

        # Provide a lay-level explanation of independent variables
        st.markdown("""
        **What are Independent Variables?**

        Independent variables, also known as predictors or features, are the factors that you believe might influence or predict the outcome you're interested in.
        These variables are used by the model to make predictions or understand relationships.

        For example, if you're trying to predict sales, independent variables might include advertising spend, number of salespeople, or market conditions.
        """)

        # Exclude identifier columns from features
        identifier_columns = ['acct_numb', 'acct_name']
        possible_features = [col for col in df.columns if col not in identifier_columns]

        # Multiselect for feature selection with a unique key
        selected_features = st.multiselect(
            "Choose your independent variables (features):",
            options=possible_features,
            default=possible_features[:3],  # Default selection
            help="Select one or more features to include in the model.",
            key='feature_selection'
        )

        if selected_features:
            st.session_state.selected_features = selected_features
            st.success(f"✅ You have selected {len(selected_features)} independent variables.")
        else:
            st.warning("⚠️ Please select at least one independent variable.")

# Step 3: Choose Model & Assign Weights
elif st.session_state.step == 3:
    st.title("💊 Behavior Prediction Platform 💊")
    st.subheader("Step 3: Choose Model & Assign Weights")
    if 'selected_features' not in st.session_state or not st.session_state.selected_features:
        st.warning("⚠️ No independent variables selected. Please go back to Step 2.")
    else:
        st.write("Select the predictive model you want to run based on your selected features.")

        df = st.session_state.df
        selected_features = st.session_state.selected_features

        # Model selection and weight assignment interface

        # Model Selection Buttons in the specified order
        col1, col2, col3 = st.columns(3)

        with col1:
            st.button("Run Linear Regression", on_click=select_linear_regression, key='model_linear_regression')
        with col2:
            st.button("Run Weighted Scoring", on_click=select_weighted_scoring, key='model_weighted_scoring')
        with col3:
            st.button("Run Prediction Modeling", on_click=select_random_forest, key='model_prediction_modeling')

        # Show description based on selected model
        if st.session_state.selected_model:
            if st.session_state.selected_model == 'linear_regression':
                st.info("**Linear Regression:** Suitable for predicting continuous values based on linear relationships.")
                st.session_state.normalized_weights = None  # Weights not needed for Linear Regression

                # Run Model Button with unique key and on_click callback
                st.button("Next →", on_click=run_selected_model, key='run_model_linear')
            else:
                if st.session_state.selected_model == 'random_forest':
                    st.info("**Prediction Modeling:** Utilize advanced algorithms to predict future outcomes based on historical data. Ideal for forecasting and handling complex patterns.")
                elif st.session_state.selected_model == 'weighted_scoring_model':
                    st.info("**Weighted Scoring Model:** Choose this model if you're looking for analysis, not prediction.")

                # Add note on top of the sliders
                st.write("**Note:** You should only custom weight your variables if you are planning to run a Weighted Scoring Model or Prediction Modeling. Linear regression does not require weighting, and you may proceed to run the model in the below step.")

                # Instructions
                st.markdown("**Assign Weights to Selected Features** 🎯")
                st.write("""
                    Assign how important each feature is in determining the outcome.
                    The weights must add up to **10**. Use the number inputs below to assign weights.
                """)

                # Initialize a dictionary to store user-assigned weights
                feature_weights = {}

                st.markdown("### 🔢 **Enter Weights for Each Feature (Total Must Be 10):**")

                # Create number inputs for each feature
                for feature in selected_features:
                    weight = st.number_input(
                        f"Weight for **{feature}**:",
                        min_value=0.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.5,
                        format="%.1f",
                        key=f"weight_input_{feature}"
                    )
                    feature_weights[feature] = weight

                # Calculate total weight
                total_weight = sum(feature_weights.values())

                # Display total weight with validation
                st.markdown("---")
                st.markdown("### 🎯 **Total Weight Assigned:**")

                if total_weight < 10:
                    status = f"❗ Total weight is **{total_weight:.1f}**, which is less than **10**."
                    color = "#FFC107"  # Yellow
                elif total_weight > 10:
                    status = f"❗ Total weight is **{total_weight:.1f}**, which is more than **10**."
                    color = "#DC3545"  # Red
                else:
                    status = f"✅ Total weight is **{total_weight:.1f}**, which meets the requirement."
                    color = "#28A745"  # Green

                # Display status
                st.markdown(
                    f"""
                    <div style="background-color:{color}; padding: 10px; border-radius: 5px;">
                        <h3 style="color:white; text-align:center;">{status}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Progress bar
                st.progress(min(total_weight / 10, 1.0))  # Progress out of 10

                # Normalize weights if necessary
                if total_weight != 10:
                    st.warning("⚠️ The total weight does not equal **10**. The weights will be normalized automatically.")
                    if total_weight > 0:
                        normalized_weights = {feature: (weight / total_weight) * 10 for feature, weight in feature_weights.items()}
                    else:
                        normalized_weights = feature_weights  # Avoid division by zero
                else:
                    normalized_weights = feature_weights

                # Display normalized weights
                st.markdown("**Normalized Weights:**")
                normalized_weights_df = pd.DataFrame({
                    'Feature': list(normalized_weights.keys()),
                    'Weight': [round(weight, 2) for weight in normalized_weights.values()]
                })
                st.dataframe(normalized_weights_df)

                # Store normalized weights in session state
                st.session_state.normalized_weights = normalized_weights

                # Run Model Button with unique key and on_click callback
                st.button("Next →", on_click=run_selected_model, key='run_model_weighted')
        else:
            st.info("Please select a model to proceed.")

# Step 4: Choose Dependent Variable (Only for Linear Regression)
elif st.session_state.step == 4:
    if st.session_state.dependent_variable_needed:
        st.title("💊 Behavior Prediction Platform 💊")
        st.subheader("Step 4: Choose Your Dependent Variable")

        df = st.session_state.df

        # Provide a lay-level definition of the dependent variable
        st.markdown("""
        **What is a Dependent Variable?**

        The dependent variable, also known as the target variable, is the main factor you're trying to understand or predict.
        It's called 'dependent' because its value depends on other variables in your data.

        For example, if you want to predict sales based on advertising spend, sales would be the dependent variable.
        """)

        # Exclude identifier columns and selected features from selection
        identifier_columns = ['acct_numb', 'acct_name']
        possible_targets = [col for col in df.columns if col not in identifier_columns + st.session_state.selected_features]

        # Dropdown for selecting the dependent variable
        target_column = st.selectbox(
            "Choose your dependent variable (the variable you want to predict):",
            options=possible_targets,
            help="Select one variable that you want to understand or predict.",
            key='target_variable_selection'
        )

        if target_column:
            st.session_state.target_column = target_column
            st.success(f"✅ You have selected **{target_column}** as your dependent variable.")

            # Now we can proceed to preprocess data with target
            preprocess_data()
            preprocess_data_with_target()

            # Run Model Button with unique key and on_click callback
            st.button("Next →", on_click=next_step, key='proceed_to_results')
    else:
        # If dependent variable is not needed, skip to results
        st.session_state.step = 5

# Step 5: Display Results
elif st.session_state.step == 5:
    st.title("💊 Behavior Prediction Platform 💊")
    st.subheader("Step 5: Your Results")

    selected_model = st.session_state.selected_model
    normalized_weights = st.session_state.normalized_weights
    df = st.session_state.df

    if 'X' not in st.session_state:
        st.error("⚠️ Required data not found. Please go back and run the model again.")
    else:
        # Execute the selected model and display results
        if selected_model == 'linear_regression':
            if 'y' not in st.session_state:
                st.error("⚠️ Dependent variable not found. Please go back and select your dependent variable.")
            else:
                with st.spinner("Training Linear Regression model..."):
                    run_linear_regression(st.session_state.X, st.session_state.y)
        elif selected_model == 'random_forest':
            if 'target_column' in st.session_state and st.session_state.target_column:
                st.session_state.y = df[st.session_state.target_column]
            else:
                st.session_state.y = df['Account Adoption Rank Order']  # Default target variable

            with st.spinner("Running Prediction Modeling..."):
                run_random_forest(st.session_state.X, st.session_state.y, normalized_weights)
        elif selected_model == 'weighted_scoring_model':
            with st.spinner("Calculating Weighted Scoring Model..."):
                run_weighted_scoring_model(df, normalized_weights, 'Account Adoption Rank Order', categorical_mappings)
        else:
            st.error("No model selected. Please go back to Step 3 and select a model.")

        # Navigation buttons on Step 5
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("← Back to Step 3", on_click=reset_to_step_3, key='back_to_step_3')
        with col2:
            st.button("🔄 Restart", on_click=reset_app, key='restart')

# Navigation buttons at the bottom with unique keys and on_click callbacks (for steps 1-4)
if st.session_state.step != 5:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.step > 1:
            st.markdown(
                """
                <style>
                .stButton>button {
                    width: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.button("← Back", on_click=prev_step, key='back_bottom')
    with col2:
        if st.session_state.step < 5:
            st.markdown(
                """
                <style>
                .stButton>button {
                    width: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.button("Next →", on_click=next_step, key='next_bottom')
