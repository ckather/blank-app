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
    st.session_state.step = 0  # Current step: 0 to 4 (Start Here to Results)

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

# Function to reset the app to Start Here
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 0

# Function to reset to Step 3 to allow running another model
def reset_to_step_3():
    st.session_state.step = 3

# Function to advance to the next step
def next_step():
    if st.session_state.step == 0:
        st.session_state.step += 1
    elif st.session_state.step == 1:
        if st.session_state.df is None:
            st.error("⚠️ Please upload a CSV file before proceeding.")
        else:
            st.session_state.step += 1
    elif st.session_state.step == 2:
        if not st.session_state.selected_features:
            st.error("⚠️ Please select at least one independent variable before proceeding.")
        else:
            st.session_state.step += 1
    elif st.session_state.step == 3:
        if st.session_state.selected_model is None:
            st.error("⚠️ Please select a model before proceeding.")
        else:
            if st.session_state.selected_model == 'linear_regression':
                if 'target_column' not in st.session_state or st.session_state.target_column is None:
                    st.error("⚠️ Please select a dependent variable before proceeding.")
                    return
                else:
                    preprocess_data()
                    preprocess_data_with_target()
            else:
                preprocess_data()
            st.session_state.step += 1
    elif st.session_state.step < 5:
        st.session_state.step += 1

# Function to go back to the previous step
def prev_step():
    if st.session_state.step > 0:
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
    # Store preprocessed data in session state for use in Step 4
    st.session_state.X = X

def preprocess_data_with_target():
    df = st.session_state.df
    target_column = st.session_state.target_column
    X = st.session_state.X
    y = df[target_column]

    y = pd.to_numeric(y, errors='coerce')
    y = y.loc[X.index]  # Align y with X after preprocessing
    st.session_state.y = y

def render_sidebar():
    """
    Renders the instructions sidebar with step highlighting.
    """
    step_titles = [
        "Start Here",
        "Step 1: Upload CSV File",
        "Step 2: Select Independent Variables",
        "Step 3: Choose Model & Assign Weights",
        "Step 4: Your Results"
    ]

    current_step = st.session_state.step

    st.sidebar.title("📖 Navigation")

    for i, title in enumerate(step_titles):
        if i == current_step:
            st.sidebar.markdown(f"### ✅ **{title}**")
        else:
            st.sidebar.markdown(f"### {title}")

    # Restart button in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.button("🔄 Restart", on_click=reset_app)

# Render the sidebar with step highlighting
render_sidebar()

# Main app logic based on current step

# Step 0: Start Here Page
if st.session_state.step == 0:
    st.title("💊 Behavior Prediction Platform 💊")
    st.markdown("""
    ### **Unlock Insights, Predict Outcomes, and Make Informed Decisions!**

    Ever wondered how certain factors influence behavior or outcomes? Whether you're looking to predict sales, understand customer behavior, or analyze trends, our platform is here to help!

    **What Can You Do with This App?**

    - **Predict Future Sales**: Use historical data to forecast future performance.
    - **Analyze Customer Behavior**: Understand which factors drive customer decisions.
    - **Rank and Score Accounts**: Prioritize accounts based on custom criteria.
    - **Explore Relationships**: See how different variables affect your target outcomes.

    **How It Works:**

    1. **Upload Your Data**: Bring in your CSV file containing the data you want to analyze.
    2. **Select Variables**: Choose the factors (independent variables) you think influence your outcome.
    3. **Choose a Model**: Pick from Linear Regression, Weighted Scoring, or Prediction Modeling.
    4. **Get Results**: View insights, charts, and download detailed reports.

    **Why Use This App?**

    - **User-Friendly**: No coding or statistical background needed!
    - **Flexible**: Tailor the analysis to your specific needs.
    - **Insightful**: Gain valuable insights to drive your strategies.

    Ready to dive in? Click **Next** to get started!
    """)

# Continue to Part 2 below.
# Continue from previous code

# Step 1: Upload CSV and Download Template
elif st.session_state.step == 1:
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
            if st.button("Run Linear Regression", key='model_linear_regression'):
                select_linear_regression()

        with col2:
            if st.button("Run Weighted Scoring", key='model_weighted_scoring'):
                select_weighted_scoring()

        with col3:
            if st.button("Run Prediction Modeling", key='model_prediction_modeling'):
                select_random_forest()

        # Show description based on selected model
        if st.session_state.selected_model:
            if st.session_state.selected_model == 'linear_regression':
                st.info("**Linear Regression:** Suitable for predicting continuous values based on linear relationships.")

                # Display selected independent variables
                st.markdown("**Selected Independent Variables:**")
                st.write(", ".join(selected_features))

                # Display dependent variable selection
                st.markdown("""
                **Select Your Dependent Variable**

                The dependent variable, also known as the target variable, is the main factor you're trying to understand or predict.
                It's called 'dependent' because its value depends on other variables in your data.
                """)

                # Exclude identifier columns and selected features from selection
                identifier_columns = ['acct_numb', 'acct_name']
                possible_targets = [col for col in df.columns if col not in identifier_columns + selected_features]

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

            else:
                if st.session_state.selected_model == 'random_forest':
                    st.info("**Prediction Modeling:** Utilize advanced algorithms to predict future outcomes based on historical data. Ideal for forecasting and handling complex patterns.")
                elif st.session_state.selected_model == 'weighted_scoring_model':
                    st.info("**Weighted Scoring Model:** Choose this model if you're looking for analysis, not prediction.")

                # Add note on top of the sliders
                st.write("**Note:** You should only custom weight your variables if you are planning to run a Weighted Scoring Model or Prediction Modeling. Linear regression does not require weighting.")

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

        else:
            st.info("Please select a model to proceed.")

# Step 4: Display Results
elif st.session_state.step == 4:
    st.title("💊 Behavior Prediction Platform 💊")
    st.subheader("Step 4: Your Results")

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
            st.session_state.y = df['Account Adoption Rank Order']  # Default target variable
            with st.spinner("Running Prediction Modeling..."):
                run_random_forest(st.session_state.X, st.session_state.y, normalized_weights)
        elif selected_model == 'weighted_scoring_model':
            with st.spinner("Calculating Weighted Scoring Model..."):
                run_weighted_scoring_model(df, normalized_weights, 'Account Adoption Rank Order', categorical_mappings)
        else:
            st.error("No model selected. Please go back to Step 3 and select a model.")

        # Navigation buttons on the Results step
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("← Back to Step 3", on_click=reset_to_step_3, key='back_to_step_3')
        with col2:
            st.button("🔄 Restart", on_click=reset_app, key='restart')

# Navigation buttons at the bottom with unique keys and on_click callbacks (for steps 0-3)
if st.session_state.step < 4:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.step > 0:
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

# Include model functions here (run_linear_regression, run_random_forest, run_weighted_scoring_model)
# Ensure they are defined after the main app logic

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

    # Display regression summary
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
        title=f'Actual vs. Predicted {y.name}'
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

    # Interpretation in layman's terms
    st.markdown("### 🔍 **Interpretation of Results:**")
    st.markdown(f"""
    - **R-squared:** Indicates that **{r_squared:.2%}** of the variability in the target variable is explained by the model.
    - **Coefficients:** A positive coefficient means that as the variable increases, the target variable tends to increase; a negative coefficient indicates an inverse relationship.
    - **P-Values:** Variables with p-values less than 0.05 are considered statistically significant.
    """)

    # Provide download link for model results
    st.markdown("### 💾 **Download Model Results**")
    # Prepare data for download
    results_df = pd.DataFrame({
        'Actual': y,
        'Predicted': predictions,
        'Residual': y - predictions
    })
    download_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=download_data,
        file_name='linear_regression_results.csv',
        mime='text/csv'
    )

def run_random_forest(X, y, normalized_weights):
    """
    Trains and evaluates a Random Forest Regressor.
    """
    st.subheader("🤖 Prediction Modeling Results")

    # Apply feature weights before training
    if normalized_weights:
        for feature, weight in normalized_weights.items():
            if feature in X.columns:
                X[feature] *= weight

    # Handle infinite values
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]  # Align y with X after dropping rows

    # Split the data into training and testing sets
    test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5, key='test_size_slider')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    st.write(f"**Training samples:** {X_train.shape[0]} | **Testing samples:** {X_test.shape[0]}")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

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
        title=f'Actual vs Predicted {y.name}'
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

    # Provide download link for model results
    st.markdown("### 💾 **Download Model Results**")
    # Prepare data for download
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions,
        'Residual': y_test - predictions
    })
    download_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=download_data,
        file_name='prediction_modeling_results.csv',
        mime='text/csv'
    )

def run_weighted_scoring_model(df, normalized_weights, target_column, mappings):
    """
    Calculates and evaluates a Weighted Scoring Model and displays the results.
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

    # Rank the accounts based on Weighted_Score
    df_encoded['Rank'] = df_encoded['Weighted_Score'].rank(method='dense', ascending=False).astype(int)

    # Divide the accounts into three groups
    df_encoded['Adopter_Category'] = pd.qcut(df_encoded['Rank'], q=3, labels=['Early Adopter', 'Middle Adopter', 'Late Adopter'])

    # Mapping adopter categories to emojis
    adopter_emojis = {
        'Early Adopter': '🚀',
        'Middle Adopter': '⏳',
        'Late Adopter': '🐢'
    }

    # Display the leaderboard
    st.markdown("### 🏆 **Leaderboard of Accounts**")
    top_n = st.slider("Select number of top accounts to display", min_value=5, max_value=50, value=10, step=1)

    top_accounts = df_encoded[['acct_numb', 'acct_name', 'Weighted_Score', 'Rank', 'Adopter_Category', target_column]].sort_values(by='Weighted_Score', ascending=False).head(top_n)

    # Display accounts with emojis
    for idx, row in top_accounts.iterrows():
        emoji = adopter_emojis.get(row['Adopter_Category'], '')
        st.markdown(f"""
        **Rank {row['Rank']}:** {emoji} **{row['acct_name']}**  
        - **Account Number:** {row['acct_numb']}  
        - **Weighted Score:** {row['Weighted_Score']:.2f}  
        - **Adopter Category:** {row['Adopter_Category']}  
        - **{target_column}:** {row[target_column]}
        """)
        st.markdown("---")

    # Provide download link for model results
    st.markdown("### 💾 **Download Model Results**")
    # Prepare data for download
    results_df = df_encoded[['acct_numb', 'acct_name', 'Weighted_Score', 'Rank', 'Adopter_Category', target_column]].sort_values(by='Weighted_Score', ascending=False)
    download_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=download_data,
        file_name='weighted_scoring_model_results.csv',
        mime='text/csv'
    )
