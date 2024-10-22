# app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import shap  # Ensure SHAP is installed: pip install shap==0.41.0

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

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

if 'X' not in st.session_state:
    st.session_state.X = None  # Preprocessed features

if 'y' not in st.session_state:
    st.session_state.y = None  # Preprocessed target

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
            if st.session_state.selected_model in ['linear_regression', 'lightgbm']:
                if 'target_column' not in st.session_state or st.session_state.target_column is None:
                    st.error("⚠️ Please select a dependent variable before proceeding.")
                    return
                else:
                    # Call the cached preprocessing functions with required arguments
                    st.session_state.X = preprocess_data_cached(st.session_state.df, st.session_state.selected_features)
                    st.session_state.y = preprocess_data_with_target_cached(st.session_state.df, st.session_state.target_column, st.session_state.X)
            elif st.session_state.selected_model == 'weighted_scoring_model':
                st.session_state.X = preprocess_data_cached(st.session_state.df, st.session_state.selected_features)
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

def select_lightgbm():
    st.session_state.selected_model = 'lightgbm'

# Define mappings for categorical features
categorical_mappings = {
    'analog_1_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_2_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_3_adopt': {'low': 1, 'medium': 2, 'high': 3}
}

# Helper Function: Encode Categorical Features
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

# Helper Function: Generate Account Adoption Rank Order
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

# Helper Function: Preprocess Data (Cached)
@st.cache_data(show_spinner=False)
def preprocess_data_cached(df, selected_features):
    """
    Preprocesses the data and caches the result to speed up the app.
    """
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
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()

    return X

# Helper Function: Preprocess Target Variable (Cached)
@st.cache_data(show_spinner=False)
def preprocess_data_with_target_cached(df, target_column, X):
    """
    Preprocesses the target variable and aligns it with X.
    """
    y = df[target_column]
    y = pd.to_numeric(y, errors='coerce')
    y = y.loc[X.index]  # Align y with X after preprocessing
    return y

# Function to render the sidebar with step highlighting
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

    # Restart button in the sidebar with a unique key
    st.sidebar.markdown("---")
    st.sidebar.button("🔄 Restart", on_click=reset_app, key='restart_sidebar')

# Model Functions

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
    st.dataframe(coef_df.style.highlight_max(axis=0, color='lightgreen'))

    st.write(f"**Coefficient of Determination (R-squared):** {r_squared:.4f}")

    # Plot Actual vs Predicted
    fig = px.scatter(
        x=y,
        y=predictions,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title=f'Actual vs. Predicted {y.name}',
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation in layman's terms
    st.markdown("### 🔍 **Key Insights:**")
    st.markdown(f"""
    - **R-squared:** The model explains **{r_squared:.2%}** of the variance in the target variable.
    - **Coefficients:** 
        - **Positive Coefficients:** Indicate a direct relationship with the target variable.
        - **Negative Coefficients:** Indicate an inverse relationship with the target variable.
    - **Statistical Significance:** Variables with p-values < 0.05 are considered significant.
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

def run_lightgbm(X, y):
    """
    Trains and evaluates a LightGBM Regressor.
    Outputs a rank-ordered list of account adoption in 2025 based on selected variables.
    """
    st.subheader("⚡ LightGBM Regression Results")

    # Inform the user about the processing time
    st.markdown("**⚠️ Note:** Training the LightGBM model may take a few minutes. Please do not refresh the page or use any navigation buttons during this process.")

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]  # Align y with X after dropping rows

    # Debugging: Check data
    st.write("**Shape of X:**", X.shape)
    st.write("**Shape of y:**", y.shape)
    st.write("**Any missing values in X?**", X.isnull().values.any())
    st.write("**Any missing values in y?**", y.isnull().values.any())

    # Check if there are enough samples after preprocessing
    if X.shape[0] < 200:
        st.error("❌ Not enough data after preprocessing to train the LightGBM model. Please ensure your dataset has at least 200 rows.")
        return

    # Split the data into training and testing sets
    test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5, key='test_size_slider_lgbm')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    st.write(f"**Training samples:** {X_train.shape[0]} | **Testing samples:** {X_test.shape[0]}")

    # Add a checkbox to allow users to skip hyperparameter tuning
    perform_hyperparameter_tuning = st.checkbox("Perform Hyperparameter Tuning for LightGBM", value=True)

    if perform_hyperparameter_tuning:
        # Optimized Hyperparameter Tuning using RandomizedSearchCV
        st.markdown("### 🔍 **Hyperparameter Tuning with Randomized Search**")
        param_grid = {
            'num_leaves': [31, 50],
            'max_depth': [10, 20],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }

        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        randomized_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        with st.spinner("🔄 Performing Hyperparameter Tuning..."):
            try:
                randomized_search.fit(X_train, y_train)
                best_lgbm = randomized_search.best_estimator_
                st.write(f"**Best Parameters:** {randomized_search.best_params_}")
            except Exception as e:
                st.error(f"❌ Error during hyperparameter tuning: {e}")
                return

        # Cross-Validation Scores
        st.markdown("### 📊 **Cross-Validation Performance**")
        try:
            cv_scores = cross_val_score(best_lgbm, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_mse = -cv_scores.mean()
            cv_std = cv_scores.std()
            st.write(f"**Cross-Validated MSE:** {cv_mse:.2f} ± {cv_std:.2f}")
        except Exception as e:
            st.error(f"❌ Error during cross-validation: {e}")
            return
    else:
        # Initialize LightGBM with default or pre-set parameters
        best_lgbm = lgb.LGBMRegressor(
            num_leaves=31,
            max_depth=10,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        with st.spinner("⚙️ Training LightGBM model with default parameters..."):
            try:
                best_lgbm.fit(X_train, y_train)
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"❌ Error during model training: {e}")
                return

    # Proceed with predictions and generate ranked list
    try:
        predictions = best_lgbm.predict(X_test)
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return

    # Evaluation Metrics
    try:
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")
    except Exception as e:
        st.error(f"❌ Error calculating evaluation metrics: {e}")
        return

    # Generate Rank-Ordered List of Account Adoption in 2025
    try:
        st.markdown("### 📋 **Rank-Ordered List of Account Adoption in 2025**")

        # Check if 'acct_numb' and 'acct_name' are in the original dataframe
        if 'acct_numb' in st.session_state.df.columns and 'acct_name' in st.session_state.df.columns:
            # Extract corresponding account identifiers from the original dataframe
            test_indices = X_test.index
            test_accounts = st.session_state.df.loc[test_indices, ['acct_numb', 'acct_name']].reset_index(drop=True)
            adoption_predictions = pd.DataFrame({
                'acct_numb': test_accounts['acct_numb'],
                'acct_name': test_accounts['acct_name'],
                'Predicted_Adoption_2025': predictions
            })
        else:
            # If account identifiers are not present, use indices
            adoption_predictions = pd.DataFrame({
                'Account_Index': X_test.index,
                'Predicted_Adoption_2025': predictions
            })

        # Sort the accounts based on predicted adoption in descending order
        adoption_predictions_sorted = adoption_predictions.sort_values(by='Predicted_Adoption_2025', ascending=False).reset_index(drop=True)

        # Assign ranks
        adoption_predictions_sorted['Rank'] = adoption_predictions_sorted.index + 1

        # Reorder columns for better display
        if 'acct_numb' in adoption_predictions_sorted.columns and 'acct_name' in adoption_predictions_sorted.columns:
            adoption_predictions_sorted = adoption_predictions_sorted[['Rank', 'acct_numb', 'acct_name', 'Predicted_Adoption_2025']]
        else:
            adoption_predictions_sorted = adoption_predictions_sorted[['Rank', 'Account_Index', 'Predicted_Adoption_2025']]

        # Add conditional formatting for better aesthetics
        def highlight_category(row):
            if row['Rank'] == 1:
                return ['background-color: gold'] * len(row)
            elif row['Rank'] == 2:
                return ['background-color: silver'] * len(row)
            elif row['Rank'] == 3:
                return ['background-color: #cd7f32'] * len(row)  # Bronze
            else:
                return [''] * len(row)

        styled_df = adoption_predictions_sorted.style.apply(highlight_category, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - **Rank:** Position of the account based on predicted adoption in 2025 (1 being the highest).
        - **Account Number / Name:** Identifier for each account.
        - **Predicted Adoption 2025:** The model's prediction of how much the account will adopt or perform in 2025 based on selected variables.
        
        **How to Use This List:**
        - **Prioritization:** Focus your efforts on higher-ranked accounts to maximize impact.
        - **Resource Allocation:** Allocate more resources to top accounts to drive better outcomes.
        - **Strategic Planning:** Use the rankings to inform your strategic decisions and marketing strategies.
        """)

    except Exception as e:
        st.error(f"❌ Error generating ranked list: {e}")
        return

    # SHAP for Model Explainability
    try:
        st.markdown("### 🧠 **Model Explainability with SHAP**")
        st.markdown("""
        **What is SHAP?**
        SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of machine learning models. It connects optimal credit allocation with local explanations using concepts from cooperative game theory.
        """)

        # Initialize SHAP explainer using TreeExplainer for LightGBM
        explainer = shap.TreeExplainer(best_lgbm)
        shap_values = explainer.shap_values(X_test)

        # SHAP Summary Plot using matplotlib for better aesthetics
        st.markdown("#### 📊 **SHAP Summary Plot**")
        fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, ax=ax_summary)
        st.pyplot(fig_summary, use_container_width=True)
        plt.clf()

        # SHAP Dependence Plot using matplotlib
        st.markdown("#### 🔍 **SHAP Dependence Plot for Top Feature**")
        # Identify the top feature based on SHAP values
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        top_feature_index = np.argmax(shap_abs_mean)
        top_feature_name = X_test.columns[top_feature_index]

        # Ensure the top feature exists and has variation
        if top_feature_name in X_test.columns:
            fig_dependence, ax_dependence = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(top_feature_name, shap_values, X_test, show=False, ax=ax_dependence)
            st.pyplot(fig_dependence, use_container_width=True)
            plt.clf()
        else:
            st.warning(f"⚠️ Top feature '{top_feature_name}' not found in the data.")

        st.markdown("""
        **How to Interpret SHAP Plots:**
        - **SHAP Summary Plot:** Shows feature importance and impact on the model's predictions.
            - **Higher SHAP Values:** Features have a strong positive impact.
            - **Lower SHAP Values:** Features have a strong negative impact.
        - **SHAP Dependence Plot:** Illustrates the relationship between a feature and the prediction.
            - **X-axis:** Feature value.
            - **Y-axis:** SHAP value (impact on prediction).
        """)

    except Exception as e:
        st.warning(f"⚠️ SHAP could not be computed: {e}. Ensure SHAP is installed correctly and compatible with your model.")
        st.warning("**Troubleshooting Steps:**")
        st.markdown("""
        1. **Ensure SHAP is Installed Correctly:**
            - Run `pip install shap==0.41.0` in your terminal.
        2. **Check SHAP Version Compatibility:**
            - Some SHAP versions may have compatibility issues with certain models.
            - Consider upgrading or downgrading SHAP as needed.
        3. **Restart the App:**
            - Sometimes, restarting the Streamlit app can resolve transient issues.
        """)

    # Provide download link for model results
    try:
        st.markdown("### 💾 **Download Model Results**")
        # Prepare data for download
        download_data = adoption_predictions_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Ranked Adoption List as CSV",
            data=download_data,
            file_name='lightgbm_ranked_adoption_2025.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"❌ Error preparing download: {e}")

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

    # Handle non-finite values in 'Weighted_Score'
    df_encoded['Weighted_Score'] = df_encoded['Weighted_Score'].replace([np.inf, -np.inf], np.nan)
    initial_row_count = df_encoded.shape[0]
    df_encoded = df_encoded.dropna(subset=['Weighted_Score'])
    final_row_count = df_encoded.shape[0]
    if final_row_count < initial_row_count:
        st.warning(f"Dropped {initial_row_count - final_row_count} rows due to non-finite 'Weighted_Score' values.")

    # Check if 'Weighted_Score' is empty after dropping NaNs
    if df_encoded.empty:
        st.error("❌ All rows have invalid 'Weighted_Score' values after preprocessing.")
        return

    # Rank the accounts based on Weighted_Score
    df_encoded['Rank'] = df_encoded['Weighted_Score'].rank(method='dense', ascending=False)

    # Ensure 'Rank' has no NaN or inf values
    df_encoded['Rank'] = df_encoded['Rank'].replace([np.inf, -np.inf], np.nan)
    df_encoded = df_encoded.dropna(subset=['Rank'])

    # Cast 'Rank' to integer
    try:
        df_encoded['Rank'] = df_encoded['Rank'].astype(int)
    except ValueError as ve:
        st.error(f"❌ Error converting 'Rank' to integer: {ve}")
        return

    # Divide the accounts into three groups
    try:
        df_encoded['Adopter_Category'] = pd.qcut(df_encoded['Rank'], q=3, labels=['Early Adopter', 'Middle Adopter', 'Late Adopter'])
    except ValueError as ve:
        st.error(f"❌ Error in categorizing adopter groups: {ve}")
        return

    # Mapping adopter categories to emojis
    adopter_emojis = {
        'Early Adopter': '🚀',
        'Middle Adopter': '⏳',
        'Late Adopter': '🐢'
    }

    # Display the leaderboard
    st.markdown("### 🏆 **Leaderboard of Accounts**")
    top_n = st.slider("Select number of top accounts to display", min_value=5, max_value=50, value=10, step=1, key='top_n_slider')

    top_accounts = df_encoded[['acct_numb', 'acct_name', 'Weighted_Score', 'Rank', 'Adopter_Category', target_column]].sort_values(by='Weighted_Score', ascending=False).head(top_n)

    # Check if top_accounts is empty
    if top_accounts.empty:
        st.warning("⚠️ No accounts to display based on the current selection.")
    else:
        # Add conditional formatting for better aesthetics
        def highlight_category(row):
            if row['Adopter_Category'] == 'Early Adopter':
                return ['background-color: lightgreen'] * len(row)
            elif row['Adopter_Category'] == 'Middle Adopter':
                return ['background-color: lightyellow'] * len(row)
            elif row['Adopter_Category'] == 'Late Adopter':
                return ['background-color: lightcoral'] * len(row)
            else:
                return [''] * len(row)

        styled_top_accounts = top_accounts.style.apply(highlight_category, axis=1)
        st.dataframe(styled_top_accounts, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - **Weighted Score:** Represents the combined weighted importance of the selected features.
        - **Rank:** Position of the account based on the weighted score.
        - **Adopter Category:** Classification into Early, Middle, or Late Adopters.
        
        **How to Use This List:**
        - **Prioritization:** Focus on Early Adopters for maximum impact.
        - **Resource Allocation:** Allocate more resources to higher-ranked accounts.
        - **Strategic Planning:** Inform your strategic and marketing decisions based on the rankings.
        """)

    # **Understanding the Scores**
    st.markdown("### 📄 **Understanding the Scores:**")
    st.markdown("""
    - **Weighted Score:** This score is calculated by multiplying each selected feature by its assigned weight and summing the results. A higher score indicates a higher priority based on your weights.
    - **Rank:** The order of accounts from highest to lowest based on the Weighted Score.
    - **Adopter Category:**
        - **Early Adopter (🚀):** Top third of accounts, highest priority.
        - **Middle Adopter (⏳):** Middle third of accounts.
        - **Late Adopter (🐢):** Bottom third of accounts, lower priority.
    """)

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
    3. **Choose a Model**: Pick from Linear Regression, Weighted Scoring, or LightGBM.
    4. **Get Results**: View insights, charts, and download detailed reports.

    **Why Use This App?**

    - **User-Friendly**: No coding or statistical background needed!
    - **Flexible**: Tailor the analysis to your specific needs.
    - **Insightful**: Gain valuable insights to drive your strategies.

    Ready to dive in? Click **Next** to get started!
    """)

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

            # Display preprocessing messages below the title
            st.write(f"**Initial data rows:** {df.shape[0]}")
            st.write(f"**Initial data columns:** {', '.join(df.columns)}")

            # Display first few rows for confirmation
            st.markdown("### 📊 **Preview of Uploaded Data:**")
            st.dataframe(df.head())

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

            # Optional: Provide feature descriptions or importance
            st.markdown("**Selected Features:**")
            for feature in selected_features:
                st.write(f"- {feature}")
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

        # Add a note about optimal data row sizes for each model
        st.markdown("""
        **📌 Optimal Data Row Recommendations:**
        - **Linear Regression:** Best suited for datasets with **10-100 rows**.
        - **LightGBM Regression:** Ideal for larger datasets with **200+ rows** as it leverages more data to train and predict effectively.
        """)

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
            if st.button("Run LightGBM", key='model_lightgbm'):
                select_lightgbm()

        # Show description based on selected model
        if st.session_state.selected_model:
            if st.session_state.selected_model in ['linear_regression', 'lightgbm']:
                if st.session_state.selected_model == 'linear_regression':
                    st.info("**Linear Regression:** Suitable for predicting continuous values based on linear relationships.")
                else:
                    st.info("**LightGBM Regression:** Utilize efficient gradient boosting to predict outcomes based on historical data. Ideal for handling large datasets with high performance.")

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

            elif st.session_state.selected_model == 'weighted_scoring_model':
                st.info("**Weighted Scoring Model:** Choose this model if you're looking for analysis, not prediction.")

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
                        st.error("❌ Total weight is zero. Please assign a positive total weight.")
                        normalized_weights = None
                else:
                    normalized_weights = feature_weights

                # Display normalized weights if normalization was successful
                if normalized_weights:
                    st.markdown("**Normalized Weights:**")
                    normalized_weights_df = pd.DataFrame({
                        'Feature': list(normalized_weights.keys()),
                        'Weight': [round(weight, 2) for weight in normalized_weights.values()]
                    })
                    st.dataframe(normalized_weights_df)

                    # Store normalized weights in session state
                    st.session_state.normalized_weights = normalized_weights
                else:
                    st.session_state.normalized_weights = None

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
        elif selected_model == 'lightgbm':
            if 'y' not in st.session_state:
                st.error("⚠️ Dependent variable not found. Please go back and select your dependent variable.")
            else:
                with st.spinner("Training LightGBM model..."):
                    run_lightgbm(st.session_state.X, st.session_state.y)
        elif selected_model == 'weighted_scoring_model':
            if normalized_weights is None:
                st.error("⚠️ Invalid weights assigned. Please go back and assign valid weights.")
            else:
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
            st.button("🔄 Restart", on_click=reset_app, key='restart_main')

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
