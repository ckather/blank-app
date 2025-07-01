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

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set the page configuration
st.set_page_config(page_title="💊 Behavior Prediction Platform 💊", layout="wide")

# Initialize session state variables for navigation and selections
if 'step' not in st.session_state:
    st.session_state.step = 0  # Current step: 0 to 5 (Start Here to Results + Demo)
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'normalized_weights' not in st.session_state:
    st.session_state.normalized_weights = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'header_row' not in st.session_state:
    st.session_state.header_row = None

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
        df['Total_2022_and_2023'] = df['ProdA_sales_2022'] + df['ProdA_sales_2023'] + df['competition_sales_2022'] + df['competition_sales_2023']
        st.info("'Total_2022_and_2023' column was missing and has been computed automatically.")
    else:
        st.info("'Total_2022_and_2023' column found in the uploaded file.")
    if 'Total_2022_and_2023' not in sales_columns:
        sales_columns.append('Total_2022_and_2023')
    missing_columns = [col for col in sales_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ The following required columns are missing to generate 'Account Adoption Rank Order': {', '.join(missing_columns)}")
        st.stop()
    df['Total_Sales'] = df[sales_columns].sum(axis=1)
    df['Account Adoption Rank Order'] = df['Total_Sales'].rank(method='dense', ascending=False).astype(int)
    return df

# Helper Function: Preprocess Data (Cached)
@st.cache_data(show_spinner=False)
def preprocess_data_cached(df, selected_features):
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

# Helper Function: Preprocess Target Variable (Cached)
@st.cache_data(show_spinner=False)
def preprocess_data_with_target_cached(df, target_column, X):
    y = df[target_column]
    y = pd.to_numeric(y, errors='coerce')
    y = y.loc[X.index]
    return y

# Function to render the sidebar with step highlighting and Demo tab
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

    # Add constant term for intercept using imported add_constant
    X = add_constant(X)

    # Fit OLS model directly
    model = OLS(y, X).fit()
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
    """
    st.subheader("⚡ LightGBM Regression Results")
    st.markdown("**⚠️ Note:** Training may take a few minutes. Please do not refresh.")

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    if X.shape[0] < 200:
        st.error("❌ Not enough data (min 200 rows).")
        return

    test_size = st.slider("Select Test Size (%)", 10, 50, 20, 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    st.write(f"**Train samples:** {X_train.shape[0]} | **Test samples:** {X_test.shape[0]}")

    perform_tuning = st.checkbox("Perform Hyperparameter Tuning", value=True)
    if perform_tuning:
        param_grid = {
            'num_leaves': [31, 50],
            'max_depth': [10, 20],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(lgbm, param_grid, n_iter=10, cv=3,
                                    scoring='neg_mean_squared_error',
                                    random_state=42, n_jobs=-1, verbose=1)
        with st.spinner("🔄 Tuning..."):
            try:
                search.fit(X_train, y_train)
                best_lgbm = search.best_estimator_
                st.write(f"**Best Params:** {search.best_params_}")
            except Exception as e:
                st.error(f"❌ Tuning error: {e}")
                return
        cv_scores = cross_val_score(best_lgbm, X_train, y_train, cv=3,
                                    scoring='neg_mean_squared_error', n_jobs=-1)
        mse, std = -cv_scores.mean(), cv_scores.std()
        st.write(f"**CV MSE:** {mse:.2f} ± {std:.2f}")
    else:
        best_lgbm = lgb.LGBMRegressor(
            num_leaves=31, max_depth=10, learning_rate=0.05,
            n_estimators=100, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        with st.spinner("⚙️ Training..."):
            try:
                best_lgbm.fit(X_train, y_train)
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"❌ Training error: {e}")
                return

    try:
        preds = best_lgbm.predict(X_test)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return

    # Metrics
    mse_val = mean_squared_error(y_test, preds)
    mae_val = mean_absolute_error(y_test, preds)
    r2_val = r2_score(y_test, preds)
    st.write(f"**MSE:** {mse_val:.2f} | **MAE:** {mae_val:.2f} | **R²:** {r2_val:.4f}")

    # Rank-Ordered List
    st.markdown("### 📋 **Rank-Ordered List**")
    if 'acct_numb' in st.session_state.df.columns and 'acct_name' in st.session_state.df.columns:
        idx = X_test.index
        accounts = st.session_state.df.loc[idx, ['acct_numb', 'acct_name']].reset_index(drop=True)
        df_preds = pd.DataFrame({
            'acct_numb': accounts['acct_numb'],
            'acct_name': accounts['acct_name'],
            'Predicted_Adoption_2025': preds
        })
    else:
        df_preds = pd.DataFrame({
            'Account_Index': X_test.index,
            'Predicted_Adoption_2025': preds
        })
    df_sorted = df_preds.sort_values(by=df_preds.columns[-1], ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1
    if 'acct_numb' in df_sorted.columns:
        df_display = df_sorted[['Rank', 'acct_numb', 'acct_name', 'Predicted_Adoption_2025']]
    else:
        df_display = df_sorted[['Rank', 'Account_Index', 'Predicted_Adoption_2025']]

    def highlight_top(row):
        if row['Rank'] == 1: return ['background-color: gold']*len(row)
        if row['Rank'] == 2: return ['background-color: silver']*len(row)
        if row['Rank'] == 3: return ['background-color: #cd7f32']*len(row)
        return ['']*len(row)

    st.dataframe(df_display.style.apply(highlight_top, axis=1), use_container_width=True)

    # SHAP Explainability
    try:
        st.markdown("### 🧠 **SHAP Explainability**")
        explainer = shap.TreeExplainer(best_lgbm)
        shap_values = explainer.shap_values(X_test)
        fig = plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig, use_container_width=True)
        plt.clf()
    except Exception as e:
        st.warning(f"⚠️ SHAP error: {e}")

    # Download ranked list
    csv_out = df_display.to_csv(index=False).encode('utf-8')
    st.download_button("Download Ranked List CSV", data=csv_out,
                       file_name='lightgbm_ranked_adoption_2025.csv',
                       mime='text/csv')

def run_weighted_scoring_model(df, normalized_weights, target_column, mappings):
    """
    Calculates and evaluates a Weighted Scoring Model and displays the results.
    """
    st.subheader("⚖️ Weighted Scoring Model Results")
    df_encoded = encode_categorical_features(df.copy(), mappings)
    df_encoded['Weighted_Score'] = 0
    for feature, weight in normalized_weights.items():
        if feature in df_encoded.columns and pd.api.types.is_numeric_dtype(df_encoded[feature]):
            df_encoded['Weighted_Score'] += df_encoded[feature] * weight
    df_encoded['Weighted_Score'] = df_encoded['Weighted_Score'].replace([np.inf, -np.inf], np.nan)
    initial = df_encoded.shape[0]
    df_encoded = df_encoded.dropna(subset=['Weighted_Score'])
    if df_encoded.shape[0] < initial:
        st.warning(f"Dropped {initial - df_encoded.shape[0]} rows due to invalid scores.")
    if df_encoded.empty:
        st.error("❌ No valid scores to rank.")
        return
    df_encoded['Rank'] = df_encoded['Weighted_Score'].rank(method='dense', ascending=False).astype(int)
    df_encoded['Adopter_Category'] = pd.qcut(df_encoded['Rank'], q=3, labels=['Early Adopter','Middle Adopter','Late Adopter'])
    emojis = {'Early Adopter':'🚀','Middle Adopter':'⏳','Late Adopter':'🐢'}

    st.markdown("### 🏆 **Leaderboard**")
    top_n = st.slider("Top N accounts", 5, 50, 10, 1)
    cols = ['acct_numb','acct_name','Weighted_Score','Rank','Adopter_Category', target_column]
    if all(c in df_encoded.columns for c in ['acct_numb','acct_name']): 
        top = df_encoded[cols].nlargest(top_n,'Weighted_Score')
    else:
        top = df_encoded[['Weighted_Score','Rank','Adopter_Category',target_column]].nlargest(top_n,'Weighted_Score').reset_index(drop=True)
        top.index += 1

    def highlight_cat(r):
        if r['Adopter_Category']=='Early Adopter': return ['background-color: lightgreen']*len(r)
        if r['Adopter_Category']=='Middle Adopter': return ['background-color: lightyellow']*len(r)
        if r['Adopter_Category']=='Late Adopter': return ['background-color: lightcoral']*len(r)
        return ['']*len(r)

    st.dataframe(top.style.apply(highlight_cat, axis=1), use_container_width=True)

    st.markdown("### 📄 **Understanding the Scores:**")
    st.markdown("""
    - **Weighted Score:** Combined weighted importance of features.
    - **Rank:** Order based on score.
    - **Adopter Category:** Early/Middle/Late Adopter.
    """)
    csv_scores = top.to_csv(index=False).encode('utf-8')
    st.download_button("Download Scoring Results CSV", data=csv_scores,
                       file_name='weighted_scoring_model_results.csv',
                       mime='text/csv')

# Demo Tab Placeholder
def run_demo():
    st.title("🎥 Demo")
    st.markdown("### **Coming Soon!** Stay tuned for a demonstration.")

# Render sidebar and main logic
render_sidebar()
if st.session_state.step == 0:
    st.title("💊 Behavior Prediction Platform 💊")
    st.markdown("""
    ### **Unlock Insights, Predict Outcomes, and Make Informed Decisions!**
    1. Upload data
    2. Select variables
    3. Choose model
    4. View results
    """)
elif st.session_state.step == 1:
    st.title("💊 Behavior Prediction Platform 💊")
    st.subheader("Step 1: Upload Your CSV/Excel File")
    st.download_button(
        label="Download CSV Template 📄",
        data=pd.DataFrame({
            'acct_numb':['123','456','789'],
            'acct_name':['A','B','C']
        }).to_csv(index=False),
        file_name='csv_template.csv',
        mime='text/csv'
    )
    uploaded_file = st.file_uploader("Choose your CSV or Excel file:", type=["csv","xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.type.endswith("sheet"):
                excel = pd.ExcelFile(uploaded_file)
                sheets = excel.sheet_names
                sheet = sheets[0] if len(sheets)==1 else st.selectbox("Select sheet:", sheets)
                df = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)
            else:
                df = pd.read_csv(uploaded_file, header=None)
            st.dataframe(df.head())
            possible = [i for i,row in df.iterrows() if row.apply(lambda x: isinstance(x,str)).sum()/len(row)>0.7]
            if len(possible)==1:
                hr = possible[0]; st.session_state.header_row=hr; st.success(f"Header at {hr}")
            elif possible:
                hr = st.selectbox("Select header row:", possible); st.session_state.header_row=hr
            else:
                hr = st.number_input("Enter header row:", 0, len(df)-1, 0); st.session_state.header_row=hr
            df = pd.read_csv(uploaded_file, header=st.session_state.header_row) if not uploaded_file.type.endswith("sheet") else pd.read_excel(uploaded_file, sheet_name=sheet, header=st.session_state.header_row)
            df = generate_account_adoption_rank(df)
            st.session_state.df = df
            st.success("File uploaded and rank generated!")
        except Exception as e:
            st.error(f"Error: {e}")
elif st.session_state.step == 2:
    st.title("💊 Behavior Prediction Platform 💊")
    df = st.session_state.df
    if df is None:
        st.warning("Please upload a file first.")
    else:
        st.subheader("Step 2: Select Independent Variables")
        id_cols = ['acct_numb','acct_name']
        features = [c for c in df.columns if c not in id_cols]
        sel = st.multiselect("Choose features:", features, default=features[:3])
        if sel:
            st.session_state.selected_features = sel
            st.success(f"Selected {len(sel)} features.")
elif st.session_state.step == 3:
    st.title("💊 Behavior Prediction Platform 💊")
    sel = st.session_state.selected_features
    if not sel:
        st.warning("Select features first.")
    else:
        st.subheader("Step 3: Choose Model & Assign Weights")
        st.markdown("**Optimal Data:** Linear:10-100 rows, LightGBM:≥200 rows")
        c1,c2,c3 = st.columns(3)
        if c1.button("Run Linear Regression"): select_linear_regression()
        if c2.button("Run Weighted Scoring"): select_weighted_scoring()
        if c3.button("Run LightGBM"): select_lightgbm()
        if st.session_state.selected_model in ['linear_regression','lightgbm']:
            st.info("Select dependent variable:")
            targets = [c for c in st.session_state.df.columns if c not in ['acct_numb','acct_name']+sel]
            tgt = st.selectbox("Dependent variable:", targets)
            if tgt:
                st.session_state.target_column = tgt; st.success(f"Target: {tgt}")
        elif st.session_state.selected_model=='weighted_scoring_model':
            st.info("Assign weights (sum 10):")
            weights = {}
            for f in sel:
                w = st.number_input(f"Weight for {f}:", 0.0, 10.0, 0.0, 0.5, key=f"w_{f}")
                weights[f]=w
            total = sum(weights.values())
            st.markdown(f"Total weight: {total}")
            if total!=10 and total>0:
                norm = {f: w/total*10 for f,w in weights.items()}
            else:
                norm = weights
            if norm:
                st.session_state.normalized_weights = norm
                st.dataframe(pd.DataFrame({'Feature':list(norm), 'Weight':[round(x,2) for x in norm.values()]}))
elif st.session_state.step == 4:
    st.title("💊 Behavior Prediction Platform 💊")
    model = st.session_state.selected_model
    if model=='linear_regression':
        run_linear_regression(st.session_state.X, st.session_state.y)
    elif model=='lightgbm':
        run_lightgbm(st.session_state.X, st.session_state.y)
    elif model=='weighted_scoring_model':
        run_weighted_scoring_model(st.session_state.df, st.session_state.normalized_weights, 'Account Adoption Rank Order', categorical_mappings)
    c1,c2 = st.columns(2)
    c1.button("← Back to Step 3", on_click=reset_to_step_3)
    c2.button("🔄 Restart", on_click=reset_app)
elif st.session_state.step == 5:
    run_demo()

if st.session_state.step < 5:
    b1,b2 = st.columns([1,1])
    if st.session_state.step>0:
        b1.button("← Back", on_click=prev_step)
    b2.button("Next →", on_click=next_step)

