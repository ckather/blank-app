# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import shap

from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="💊 Behavior Prediction Platform 💊", layout="wide")

# Always render sidebar
def render_sidebar():
    step_titles = [
        "Start Here",
        "Step 1: Upload CSV/Excel File",
        "Step 2: Select Independent Variables",
        "Step 3: Choose Model & Assign Weights",
        "Step 4: Your Results",
        "Demo"
    ]
    current_step = st.session_state.get("step", 0)
    st.sidebar.title("📖 Navigation")
    for i, title in enumerate(step_titles):
        if i == current_step:
            st.sidebar.markdown(f"### ✅ **{title}**")
        else:
            st.sidebar.markdown(f"### {title}")
    st.sidebar.markdown("---")
    st.sidebar.button("🔄 Restart", on_click=lambda: reset_app(), key='restart_sidebar')

render_sidebar()

# Session state defaults
defaults = {
    'step': 0,
    'df': None,
    'target_column': None,
    'selected_features': [],
    'selected_model': None,
    'normalized_weights': None,
    'X': None,
    'y': None,
    'header_row': None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    for key, val in defaults.items():
        st.session_state[key] = val

def reset_to_step_3():
    st.session_state.step = 3

def next_step():
    step = st.session_state.step
    if step == 0:
        st.session_state.step += 1
    elif step == 1 and st.session_state.df is not None:
        st.session_state.step += 1
    elif step == 2 and st.session_state.selected_features:
        st.session_state.step += 1
    elif step == 3:
        if st.session_state.selected_model is None:
            st.error("⚠️ Please select a model.")
        elif st.session_state.selected_model in ['linear_regression', 'lightgbm'] and not st.session_state.target_column:
            st.error("⚠️ Please select a dependent variable.")
        else:
            df = st.session_state.df
            st.session_state.X = preprocess_data_cached(df.copy(), st.session_state.selected_features)
            if st.session_state.selected_model != 'weighted_scoring_model':
                st.session_state.y = preprocess_data_with_target_cached(df.copy(), st.session_state.target_column, st.session_state.X)
            st.session_state.step += 1
    elif step < 5:
        st.session_state.step += 1

def prev_step():
    st.session_state.step = max(0, st.session_state.step - 1)

# Model selectors
def select_linear_regression():
    st.session_state.selected_model = 'linear_regression'

def select_weighted_scoring():
    st.session_state.selected_model = 'weighted_scoring_model'

def select_lightgbm():
    st.session_state.selected_model = 'lightgbm'

# Continue to Part 2...
# Categorical mappings
categorical_mappings = {
    'analog_1_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_2_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_3_adopt': {'low': 1, 'medium': 2, 'high': 3}
}

# Helper functions
def encode_categorical_features(df, mappings):
    for feature, mapping in mappings.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping)
            if df[feature].isnull().any():
                df[feature].fillna(df[feature].mode()[0], inplace=True)
    return df

def generate_account_adoption_rank(df):
    sales_cols = [
        'ProdA_sales_first12', 'ProdA_sales_2022', 'ProdA_sales_2023',
        'competition_sales_first12', 'competition_sales_2022', 'competition_sales_2023'
    ]
    if 'Total_2022_and_2023' not in df.columns:
        missing = [col for col in ['ProdA_sales_2022', 'ProdA_sales_2023', 'competition_sales_2022', 'competition_sales_2023'] if col not in df.columns]
        if missing:
            st.error(f"Missing columns to compute Total_2022_and_2023: {missing}")
            st.stop()
        df['Total_2022_and_2023'] = df['ProdA_sales_2022'] + df['ProdA_sales_2023'] + df['competition_sales_2022'] + df['competition_sales_2023']
    if 'Total_2022_and_2023' not in sales_cols:
        sales_cols.append('Total_2022_and_2023')
    df['Total_Sales'] = df[sales_cols].sum(axis=1, min_count=1)
    df['Account Adoption Rank Order'] = df['Total_Sales'].rank(method='dense', ascending=False).astype(int)
    return df

@st.cache_data(show_spinner=False)
def preprocess_data_cached(df, selected_features):
    df = df.copy()
    X = df[selected_features]
    X = pd.get_dummies(X, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].mean(), inplace=True)
    return X.dropna()

@st.cache_data(show_spinner=False)
def preprocess_data_with_target_cached(df, target_column, X):
    df = df.copy()
    y = pd.to_numeric(df[target_column], errors='coerce')
    return y.loc[X.index]

# === Main App Logic ===

if st.session_state.step == 0:
    st.title("💊 Behavior Prediction Platform 💊")
    st.markdown("Start by uploading your data.")

elif st.session_state.step == 1:
    st.subheader("Step 1: Upload CSV/Excel File")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            file_type = uploaded_file.name.lower()
            df_preview = pd.read_excel(uploaded_file, header=None) if file_type.endswith('.xlsx') else pd.read_csv(uploaded_file, header=None)
            st.dataframe(df_preview.head())

            header_row = st.number_input("Enter header row index:", 0, len(df_preview) - 1, 0)
            uploaded_file = BytesIO(uploaded_file.getvalue())
            uploaded_file.seek(0)

            df = pd.read_excel(uploaded_file, header=header_row) if file_type.endswith('.xlsx') else pd.read_csv(uploaded_file, header=header_row)
            df = generate_account_adoption_rank(df)
            st.session_state.df = df
            st.success("✅ File uploaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

elif st.session_state.step == 2:
    st.subheader("Step 2: Select Independent Variables")
    df = st.session_state.df
    columns = [col for col in df.columns if col not in ['acct_numb', 'acct_name']]
    selected = st.multiselect("Select features:", columns)
    if selected:
        st.session_state.selected_features = selected
        st.success("✅ Features selected.")

elif st.session_state.step == 3:
    st.subheader("Step 3: Choose Model & Assign Weights")
    sel = st.session_state.selected_features
    df = st.session_state.df
    if not sel:
        st.warning("⚠️ Please select features first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Run Linear Regression"):
                select_linear_regression()
        with col2:
            if st.button("Run Weighted Scoring"):
                select_weighted_scoring()
        with col3:
            if st.button("Run LightGBM"):
                select_lightgbm()

        if st.session_state.selected_model in ['linear_regression', 'lightgbm']:
            targets = [c for c in df.columns if c not in ['acct_numb', 'acct_name'] + sel]
            tgt = st.selectbox("Select target variable:", options=targets)
            st.session_state.target_column = tgt
        elif st.session_state.selected_model == 'weighted_scoring_model':
            weights = {}
            for feature in sel:
                weights[feature] = st.number_input(f"Weight for {feature}:", 0.0, 10.0, 0.0, 0.5)
            total = sum(weights.values())
            if total == 0:
                st.error("⚠️ Total weight must be greater than 0.")
            else:
                normalized = {f: (w / total) * 10 for f, w in weights.items()}
                st.session_state.normalized_weights = normalized
                st.dataframe(pd.DataFrame({
                    "Feature": normalized.keys(),
                    "Weight": [round(v, 2) for v in normalized.values()]
                }))
elif st.session_state.step == 4:
    st.subheader("Step 4: Results")
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

    col1, col2 = st.columns(2)
    with col1:
        st.button("← Back to Step 3", on_click=reset_to_step_3)
    with col2:
        st.button("🔄 Restart", on_click=reset_app)

elif st.session_state.step == 5:
    st.title("🎥 Demo")
    st.markdown("Demo coming soon.")

# Navigation Buttons
if st.session_state.step < 5:
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.step > 0:
            st.button("← Back", on_click=prev_step)
    with col2:
        st.button("Next →", on_click=next_step)

# --- MODEL FUNCTION STUBS (Insert full logic later) ---
def run_linear_regression(X, y):
    st.write("📈 Linear Regression ran (stub).")

def run_lightgbm(X, y):
    st.write("⚡ LightGBM ran (stub).")
    try:
        explainer = shap.Explainer(lambda X: np.random.rand(X.shape[0]), X)
        shap_values = explainer(X)
        st.pyplot(shap.summary_plot(shap_values, X, plot_type="bar", show=False))
    except Exception as e:
        st.warning(f"SHAP plot failed: {e}")

def run_weighted_scoring_model(df, weights, target_column, mappings):
    st.write("⚖️ Weighted Scoring ran (stub).")
    df = encode_categorical_features(df.copy(), mappings)
    df["score"] = 0
    for f, w in weights.items():
        if f in df.columns:
            df["score"] += df[f] * w
    df["Rank"] = df["score"].rank(ascending=False).astype(int)
    st.dataframe(df.sort_values("score", ascending=False).head(10))


