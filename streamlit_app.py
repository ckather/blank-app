# app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np  # Added import for NumPy
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set the page configuration
st.set_page_config(page_title="💊 Pathways Prediction Platform", layout="wide")

# Initialize session state variables for navigation and selections
if 'step' not in st.session_state:
    st.session_state.step = 1  # Current step: 1 to 4

if 'df' not in st.session_state:
    st.session_state.df = None  # Uploaded DataFrame

if 'target_column' not in st.session_state:
    st.session_state.target_column = 'Account Adoption Rank Order'  # Default target variable

if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []  # Selected independent variables

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None  # Selected model

if 'model_has_run' not in st.session_state:
    st.session_state.model_has_run = False  # Indicates if a model has been run

# Function to reset the app to Step 1
def reset_app():
    st.session_state.step = 1
    st.session_state.df = None
    st.session_state.target_column = 'Account Adoption Rank Order'
    st.session_state.selected_features = []
    st.session_state.selected_model = None
    st.session_state.model_has_run = False
    st.experimental_rerun()

# Function to advance to the next step
def next_step():
    if st.session_state.step == 1 and st.session_state.df is None:
        st.warning("⚠️ Please upload a CSV file before proceeding.")
    elif st.session_state.step < 4:
        st.session_state.step += 1

# Function to go back to the previous step
def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1

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
    # Define the columns to sum for ranking (adjust these columns based on your data)
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

def run_linear_regression(X, y):
    """
    Trains and evaluates a Linear Regression model using statsmodels.
    """
    st.subheader("📈 Linear Regression Results")

    # Convert data to numeric, drop rows with NaN values
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(y.name, axis=1)
    y = data[y.name]

    # Add constant term for intercept
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    st.write("**Regression Summary:**")
    st.text(model.summary())

    # Extract R-squared
    r_squared = model.rsquared

    # Create DataFrame for coefficients
    coef_df = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std. Error': model.bse.values,
        'P-Value': model.pvalues.values
    })

    st.write("**Coefficients:**")
    st.dataframe(coef_df)

    st.write(f"**Coefficient of Determination (R-squared):** {r_squared:.4f}")

    # Plot Actual vs Predicted
    fig = px.scatter(
        x=y,
        y=predictions,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title='Actual vs Predicted'
    )
    fig.add_shape(
        type="line",
        x0=y.min(),
        y0=y.min(),
        x1=y.max(),
        y1=y.max(),
        line=dict(color="Red", dash="dash")
    )
    st.plotly_chart(fig)

def run_random_forest(X, y, normalized_weights):
    """
    Trains and evaluates a Random Forest Regressor.
    """
    # Apply feature weights before training
    if normalized_weights:
        for feature, weight in normalized_weights.items():
            if feature in X.columns:
                X[feature] *= weight

    # Handle infinite values
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    X = X.dropna()

    # Split the data into training and testing sets
    test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5, key='test_size_slider')
    X_train, X_test, y_train, y_test = train_test_split(X, y.loc[X.index], test_size=test_size/100, random_state=42)

    st.write(f"**Training samples:** {X_train.shape[0]} | **Testing samples:** {X_test.shape[0]}")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    st.subheader("🌲 Random Forest Results")
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
    fig2 = px.scatter(
        x=y_test,
        y=predictions,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title='Actual vs Predicted'
    )
    fig2.add_shape(
        type="line",
        x0=y_test.min(),
        y0=y_test.min(),
        x1=y_test.max(),
        y1=y_test.max(),
        line=dict(color="Red", dash="dash")
    )
    st.plotly_chart(fig2)

def run_weighted_scoring_model(df, normalized_weights, target_column, mappings):
    """
    Calculates and evaluates a Weighted Scoring Model based on selected features and their normalized weights.
    """
    st.subheader("⚖️ Weighted Scoring Model Results")

    # Encode categorical features
    df_encoded = encode_categorical_features(df.copy(), mappings)

    # Calculate weighted score based on normalized weights
    df_encoded['Weighted_Score'] = 0
    for feature, weight in normalized_weights.items():
        if feature in df_encoded.columns:
            if pd.api.types.is_numeric_dtype(df_encoded[feature]):
                df_encoded['Weighted_Score'] += df_encoded[feature] * weight
            else:
                st.warning(f"Feature '{feature}' is not numeric and will be skipped in scoring.")
        else:
            st.warning(f"Feature '{feature}' not found in the data and will be skipped.")

    # Correlation with target
    correlation = df_encoded['Weighted_Score'].corr(df_encoded[target_column])
    st.write(f"**Correlation between Weighted Score and {target_column}:** {correlation:.2f}")

    # Display top accounts based on score
    top_n = st.slider("Select number of top accounts to display", min_value=5, max_value=20, value=10, step=1)
    top_accounts = df_encoded[['acct_numb', 'acct_name', 'Weighted_Score', target_column]].sort_values(by='Weighted_Score', ascending=False).head(top_n)
    st.dataframe(top_accounts)

    # Plot Weighted Score vs Target
    fig = px.scatter(
        df_encoded,
        x='Weighted_Score',
        y=target_column,
        labels={'Weighted_Score': 'Weighted Score', target_column: target_column},
        title='Weighted Score vs Actual'
    )
    st.plotly_chart(fig)

def run_selected_model(normalized_weights):
    """
    Executes the selected model based on user input.
    """
    df = st.session_state.df
    target_column = st.session_state.target_column
    selected_features = st.session_state.selected_features
    selected_model = st.session_state.selected_model

    # Preprocess data
    X = df[selected_features].copy()
    y = df[target_column]

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
    y = y.loc[X.index]  # Align y with X after dropping rows

    # Execute selected model with loading spinner
    if selected_model == 'linear_regression':
        with st.spinner("Training Linear Regression model..."):
            run_linear_regression(X, y)
    elif selected_model == 'random_forest':
        with st.spinner("Training Random Forest model..."):
            run_random_forest(X, y, normalized_weights)
    elif selected_model == 'weighted_scoring_model':
        with st.spinner("Calculating Weighted Scoring Model..."):
            run_weighted_scoring_model(df, normalized_weights, target_column, categorical_mappings)

    # After running the model, set model_has_run to True
    st.session_state.model_has_run = True

    # After running the model, reset selected_model to allow re-selection
    st.session_state.selected_model = None

def render_sidebar():
    """
    Renders the instructions sidebar with step highlighting.
    """
    step_titles = ["Upload CSV File", "Confirm Target Variable", "Select Independent Variables", "Choose Model & Assign Weights"]
    current_step = st.session_state.step

    st.sidebar.title("📖 Instructions")

    for i, title in enumerate(step_titles, 1):
        if i == current_step:
            # Highlight current step with green checkmark
            st.sidebar.markdown(f"### ✅ **Step {i}: {title}**")
        elif i < current_step:
            # Completed steps with green checkmark
            st.sidebar.markdown(f"### ✅ Step {i}: {title}")
        else:
            # Upcoming steps
            st.sidebar.markdown(f"### Step {i}: {title}")

# Render the sidebar with step highlighting
render_sidebar()

# Main app logic based on current step

# Step 1: Upload CSV and Download Template
if st.session_state.step == 1:
    st.title("💊 Pathways Prediction Platform")
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

# Step 2: Confirm Target Variable
elif st.session_state.step == 2:
    st.subheader("Step 2: Confirm Target Variable")

    # Display the selected target variable
    target_column = st.session_state.target_column if st.session_state.target_column else 'Account Adoption Rank Order'
    st.session_state.target_column = target_column  # Ensure it's set

    st.markdown(f"**Selected Target Variable:** `{target_column}`")

    # Add descriptive text guiding to next steps
    st.write("""
        In the next step, you will select the independent variables that will be used to predict the **Account Adoption Rank Order**.
        This guided process ensures that you choose the most relevant features for accurate predictions.
    """)

# Step 3: Select Independent Variables
elif st.session_state.step == 3:
    df = st.session_state.df
    target_column = st.session_state.target_column
    st.subheader("Step 3: Select Independent Variables")

    # Exclude target and identifier columns from features
    identifier_columns = ['acct_numb', 'acct_name']
    possible_features = [col for col in df.columns if col not in [target_column] + identifier_columns]

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
    else:
        st.warning("⚠️ Please select at least one independent variable.")

# Step 4: Choose Model & Assign Weights
elif st.session_state.step == 4:
    df = st.session_state.df
    target_column = st.session_state.target_column
    selected_features = st.session_state.selected_features

    st.subheader("Step 4: Choose Model & Assign Weights")
    st.write("Select the predictive model you want to run based on your selected features.")

    # Model Selection Buttons in the specified order
    col1, col2, col3 = st.columns(3)

    with col1:
        st.button("Run Linear Regression", on_click=lambda: setattr(st.session_state, 'selected_model', 'linear_regression'), key='model_linear_regression')
    with col2:
        st.button("Run Weighted Scoring", on_click=lambda: setattr(st.session_state, 'selected_model', 'weighted_scoring_model'), key='model_weighted_scoring')
    with col3:
        st.button("Run Random Forest", on_click=lambda: setattr(st.session_state, 'selected_model', 'random_forest'), key='model_random_forest')

    # Show description based on selected model
    if st.session_state.selected_model:
        if st.session_state.selected_model == 'linear_regression':
            st.info("**Linear Regression:** Suitable for predicting continuous values based on linear relationships. Note: Linear regression does not require weighting, and you may proceed to run the model in the below step.")
            normalized_weights = None  # Weights not needed for Linear Regression
        else:
            if st.session_state.selected_model == 'random_forest':
                st.info("**Random Forest:** Ideal for handling complex datasets with non-linear relationships and interactions.")
            elif st.session_state.selected_model == 'weighted_scoring_model':
                st.info("**Weighted Scoring Model:** Choose this model if you're looking for analysis, not prediction.")

            # Add note on top of the sliders
            st.write("**Note:** You should only custom weight your variables if you are planning to run a Weighted Scoring Model or a Random Forest. Linear regression does not require weighting, and you may proceed to run the model in the below step.")

            # Instructions
            st.markdown("**Assign Weights to Selected Features** 🎯")
            st.write("""
                Assign how important each feature is in determining the **Account Adoption Rank Order**. 
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

        # Run Model Button with unique key and on_click callback
        st.button("Run Model", on_click=lambda: run_selected_model(normalized_weights), key='run_model')

        # Display "Run a New Model" button after the model has run
        if st.session_state.model_has_run:
            # Display "Run a New Model" button
            st.button("Run a New Model 🔄", on_click=reset_app, key='reset_app_bottom')

# Navigation buttons at the bottom with unique keys and on_click callbacks
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
    if st.session_state.step < 4:
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
