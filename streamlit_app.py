# app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Replace broad statsmodels.api import with only the components we need:
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import shap  # Ensure SHAP is installed: pip install shap==0.41.0

# ✅ FIX 1: Add BytesIO to enable proper file rewinding
from io import BytesIO

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set the page configuration
st.set_page_config(page_title="💊 Behavior Prediction Platform 💊", layout="wide")

# Initialize session state variables for navigation and selections
if 'step' not in st.session_state:
    st.session_state.step = 0  # Current step: 0 to 5 (Start Here to Results + Demo)
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
if 'header_row' not in st.session_state:
    st.session_state.header_row = None  # Row number containing headers

# Function to reset the app to Start Here
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 0

# Function to reset to Step 3 to allow running another model
def reset_to_step_3():
    st.session_state.step = 3

render_sidebar()


# Function to advance to the next step
def next_step():
    if st.session_state.step == 0:
        st.session_state.step += 1
    elif st.session_state.step == 1:
        if st.session_state.df is None:
            st.error("⚠️ Please upload a CSV or Excel file before proceeding.")
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
            return
        if st.session_state.selected_model in ['linear_regression', 'lightgbm']:
            if not st.session_state.target_column:
                st.error("⚠️ Please select a dependent variable before proceeding.")
                return
            st.session_state.X = preprocess_data_cached(st.session_state.df, st.session_state.selected_features)
            st.session_state.y = preprocess_data_with_target_cached(
                st.session_state.df, st.session_state.target_column, st.session_state.X
            )
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
    for feature, mapping in mappings.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping)
            if df[feature].isnull().any():
                st.warning(f"Some values in '{feature}' couldn't be mapped and are set to NaN.")
                df[feature].fillna(df[feature].mode()[0], inplace=True)
    return df

# Helper Function: Generate Account Adoption Rank Order
def generate_account_adoption_rank(df):
    sales_columns = [
        'ProdA_sales_first12',
        'ProdA_sales_2022',
        'ProdA_sales_2023',
        'competition_sales_first12',
        'competition_sales_2022',
        'competition_sales_2023'
    ]

    if 'Total_2022_and_2023' not in df.columns:
        required_for_total = ['ProdA_sales_2022', 'ProdA_sales_2023', 'competition_sales_2022', 'competition_sales_2023']
        missing_total_cols = [col for col in required_for_total if col not in df.columns]
        if missing_total_cols:
            st.error(f"❌ To compute 'Total_2022_and_2023', the following columns are missing: {', '.join(missing_total_cols)}")
            st.stop()
        df['Total_2022_and_2023'] = (
            df['ProdA_sales_2022'] + df['ProdA_sales_2023'] +
            df['competition_sales_2022'] + df['competition_sales_2023']
        )
        st.info("'Total_2022_and_2023' column was missing and has been computed automatically.")
    else:
        st.info("'Total_2022_and_2023' column found in the uploaded file.")

    if 'Total_2022_and_2023' not in sales_columns:
        sales_columns.append('Total_2022_and_2023')

    missing_columns = [col for col in sales_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns to generate 'Account Adoption Rank Order': {', '.join(missing_columns)}")
        st.stop()

    df['Total_Sales'] = df[sales_columns].sum(axis=1)
    df['Account Adoption Rank Order'] = df['Total_Sales'].rank(method='dense', ascending=False).astype(int)
    return df

# ✅ FIX 3: Safe preprocessing functions
@st.cache_data(show_spinner=False)
def preprocess_data_cached(df, selected_features):
    df = df.copy()
    X = df[selected_features].copy()
    selected_categorical = [col for col in selected_features if df[col].dtype == 'object']
    if selected_categorical:
        X = pd.get_dummies(X, columns=selected_categorical, drop_first=True)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].mean(), inplace=True)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    return X

@st.cache_data(show_spinner=False)
def preprocess_data_with_target_cached(df, target_column, X):
    df = df.copy()
    y = df[target_column]
    y = pd.to_numeric(y, errors='coerce')
    y = y.loc[X.index]
    return y

# Function to render the sidebar
def render_sidebar():
    step_titles = [
        "Start Here",
        "Step 1: Upload CSV/Excel File",
        "Step 2: Select Independent Variables",
        "Step 3: Choose Model & Assign Weights",
        "Step 4: Your Results",
        "Demo"
    ]
    current_step = st.session_state.step
    st.sidebar.title("📖 Navigation")
    for i, title in enumerate(step_titles):
        if i == current_step:
            st.sidebar.markdown(f"### ✅ **{title}**")
        else:
            st.sidebar.markdown(f"### {title}")
    st.sidebar.markdown("---")
    st.sidebar.button("🔄 Restart", on_click=reset_app, key='restart_sidebar')

# --------------------------------------------------------
# MODEL FUNCTIONS — MUST BE DEFINED BEFORE THEY'RE CALLED
# --------------------------------------------------------

def run_linear_regression(X, y):
    st.subheader("📈 Linear Regression Results")
    st.write("This is a placeholder. Replace with full linear regression logic.")

def run_lightgbm(X, y):
    st.subheader("⚡ LightGBM Regression Results")
    st.write("This is a placeholder. Replace with full LightGBM logic.")

def run_weighted_scoring_model(df, normalized_weights, target_column, mappings):
    st.subheader("⚖️ Weighted Scoring Model Results")
    st.write("This is a placeholder. Replace with full weighted scoring logic.")


# Step 0: Introduction screen
if st.session_state.step == 0:
    st.title("💊 Behavior Prediction Platform 💊")
    st.markdown("""
    ### **Unlock Insights, Predict Outcomes, and Make Informed Decisions!**
    1. Upload Your Data  
    2. Select Variables  
    3. Choose Model  
    4. View Results
    """)

# Step 1: File upload
elif st.session_state.step == 1:
    st.title("💊 Behavior Prediction Platform 💊")
    st.subheader("Step 1: Upload Your CSV/Excel File")

    # Template download
    st.download_button(
        label="📄 Download CSV Template",
        data=pd.DataFrame({
            'acct_numb': ['123', '456', '789'],
            'acct_name': ['Account A', 'Account B', 'Account C'],
            'ProdA_sales_first12': [10000, 15000, 12000],
            'competition_sales_2022': [5000, 6000, 5500],
            'ProdA_sales_2022': [20000, 25000, 22000],
            'ProdA_sales_2023': [30000, 35000, 32000],
            'competition_sales_2023': [15000, 16000, 15500],
            'analog_1_adopt': ['low', 'medium', 'high']
        }).to_csv(index=False),
        file_name='csv_template.csv',
        mime='text/csv'
    )

    uploaded_file = st.file_uploader("Choose your CSV or Excel file:", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # ✅ FIX 5: safer file type check
            file_type = uploaded_file.name.lower()

            # First read to detect headers
            if file_type.endswith('.xlsx'):
                excel = pd.ExcelFile(uploaded_file)
                sheets = excel.sheet_names
                df_preview = pd.read_excel(uploaded_file, sheet_name=sheets[0], header=None)
            else:
                df_preview = pd.read_csv(uploaded_file, header=None)

            st.markdown("### 📊 Preview of Uploaded Data:")
            st.dataframe(df_preview.head())

            # Detect possible header rows
            possible_header_rows = []
            for idx, row in df_preview.iterrows():
                text_count = row.apply(lambda x: isinstance(x, str)).sum()
                if len(row) and (text_count / len(row)) > 0.7:
                    possible_header_rows.append(idx)

            if possible_header_rows:
                header_row = (
                    possible_header_rows[0] if len(possible_header_rows) == 1 else
                    st.selectbox("Select header row:", options=possible_header_rows, key='header_row_selection')
                )
                st.session_state.header_row = header_row
                st.success(f"✅ Header row set to index {header_row}")
            else:
                header_row = st.number_input(
                    "Enter header row index (0-indexed):", min_value=0,
                    max_value=len(df_preview) - 1, value=0, step=1, key='manual_header_row'
                )
                st.session_state.header_row = header_row
                st.info(f"⚠️ Using header row {header_row}")

            # ✅ FIX 1: rewind the uploaded file before second read
            uploaded_file = BytesIO(uploaded_file.getvalue())
            uploaded_file.seek(0)

            # Now read with proper header row
            if file_type.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, sheet_name=sheets[0], header=st.session_state.header_row)
            else:
                df = pd.read_csv(uploaded_file, header=st.session_state.header_row)

            df = generate_account_adoption_rank(df)
            st.session_state.df = df
            st.success("✅ File uploaded and processed successfully.")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")
# Step 2: Feature selection
elif st.session_state.step == 2:
    st.title("💊 Behavior Prediction Platform 💊")
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ No data found. Please go back to Step 1 and upload your file.")
    else:
        st.subheader("Step 2: Select Independent Variables")
        identifier_columns = ['acct_numb', 'acct_name']
        possible_features = [col for col in df.columns if col not in identifier_columns]
        selected = st.multiselect(
            "Choose your independent variables (features):",
            options=possible_features,
            default=possible_features[:3],
            key='feature_selection'
        )
        if selected:
            st.session_state.selected_features = selected
            st.success(f"✅ You have selected {len(selected)} independent variables.")

# Step 3: Model selection and configuration
elif st.session_state.step == 3:
    st.title("💊 Behavior Prediction Platform 💊")
    sel = st.session_state.selected_features
    if not sel:
        st.warning("⚠️ Please select features first.")
    else:
        st.subheader("Step 3: Choose Model & Assign Weights")
        st.markdown("**Optimal Data Rows:** Linear 10–100, LightGBM ≥200")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Run Linear Regression", key='model_lr'):
                select_linear_regression()
        with c2:
            if st.button("Run Weighted Scoring", key='model_ws'):
                select_weighted_scoring()
        with c3:
            if st.button("Run LightGBM", key='model_lgb'):
                select_lightgbm()

        if st.session_state.selected_model in ['linear_regression', 'lightgbm']:
            st.info("Select your dependent variable:")
            targets = [c for c in df.columns if c not in ['acct_numb','acct_name']+sel]
            tgt = st.selectbox("Choose dependent variable:", options=targets, key='target_variable_selection')
            if tgt:
                st.session_state.target_column = tgt
                st.success(f"✅ You have selected **{tgt}** as your dependent variable.")
        elif st.session_state.selected_model == 'weighted_scoring_model':
            st.info("Assign weights (sum must equal 10):")
            feature_weights = {}
            for feature in sel:
                w = st.number_input(f"Weight for **{feature}**:", 0.0, 10.0, 0.0, 0.5, key=f"weight_{feature}")
                feature_weights[feature] = w
            total_weight = sum(feature_weights.values())
            st.markdown(f"**Total weight:** {total_weight}")
            
            # ✅ FIX 4: Guard against division by zero
            if total_weight == 0:
                st.error("⚠️ Total weight must be greater than 0.")
            else:
                normalized = {f: (w / total_weight) * 10 for f, w in feature_weights.items()}
                st.session_state.normalized_weights = normalized
                st.dataframe(pd.DataFrame({
                    'Feature': list(normalized.keys()),
                    'Weight': [round(v, 2) for v in normalized.values()]
                }))
# Step 4: Show results
elif st.session_state.step == 4:
    st.title("💊 Behavior Prediction Platform 💊")
    model = st.session_state.selected_model

    if model == 'linear_regression':
        run_linear_regression(st.session_state.X, st.session_state.y)

    elif model == 'lightgbm':
        run_lightgbm(st.session_state.X, st.session_state.y)

    elif model == 'weighted_scoring_model':
        run_weighted_scoring_model(
            st.session_state.df,
            st.session_state.normalized_weights,
            'Account Adoption Rank Order',
            categorical_mappings
        )

    c1, c2 = st.columns(2)
    with c1:
        st.button("← Back to Step 3", on_click=reset_to_step_3)
    with c2:
        st.button("🔄 Restart", on_click=reset_app)

# Step 5: Demo placeholder
elif st.session_state.step == 5:
    st.title("🎥 Demo")
    st.markdown("""
    ### **Coming Soon!**
    Stay tuned for a comprehensive demonstration of the **💊 Behavior Prediction Platform 💊**.
    """)

# Navigation buttons (bottom of app)
if st.session_state.step < 5:
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.session_state.step > 0:
            st.button("← Back", on_click=prev_step)
    with b2:
        st.button("Next →", on_click=next_step)

# ✅ FIX 2: Wrapped SHAP summary plot in try/except is already inside run_lightgbm()
# Ensure your run_lightgbm() definition has this:

# try:
#     explainer = shap.TreeExplainer(best_lgbm)
#     shap_values = explainer.shap_values(X_test)
#     fig_summary = plt.figure(figsize=(10, 6))
#     shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
#     st.pyplot(fig_summary, use_container_width=True)
#     plt.clf()
# except Exception as e:
#     st.warning(f"⚠️ SHAP error: {e}")

