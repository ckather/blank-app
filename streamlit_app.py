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
    st.session_state.step = 0
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
            st.session_state.X = preprocess_data_cached(
                st.session_state.df,
                st.session_state.selected_features
            )
            st.session_state.y = preprocess_data_with_target_cached(
                st.session_state.df,
                st.session_state.target_column,
                st.session_state.X
            )
        elif st.session_state.selected_model == 'weighted_scoring_model':
            st.session_state.X = preprocess_data_cached(
                st.session_state.df,
                st.session_state.selected_features
            )
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
        required = ['ProdA_sales_2022', 'ProdA_sales_2023', 'competition_sales_2022', 'competition_sales_2023']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns for Total_2022_and_2023: {', '.join(missing)}")
            st.stop()
        df['Total_2022_and_2023'] = (
            df['ProdA_sales_2022'] +
            df['ProdA_sales_2023'] +
            df['competition_sales_2022'] +
            df['competition_sales_2023']
        )
        st.info("'Total_2022_and_2023' computed automatically.")
    if 'Total_2022_and_2023' not in sales_columns:
        sales_columns.append('Total_2022_and_2023')
    missing = [c for c in sales_columns if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns for ranking: {', '.join(missing)}")
        st.stop()
    df['Total_Sales'] = df[sales_columns].sum(axis=1)
    df['Account Adoption Rank Order'] = df['Total_Sales'].rank(
        method='dense', ascending=False
    ).astype(int)
    return df

# Helper Function: Preprocess Data (Cached)
@st.cache_data(show_spinner=False)
def preprocess_data_cached(df, selected_features):
    X = df[selected_features].copy()
    cats = [c for c in selected_features if df[c].dtype == 'object']
    if cats:
        X = pd.get_dummies(X, columns=cats, drop_first=True)
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

# Function to render the sidebar
def render_sidebar():
    titles = [
        "Start Here",
        "Step 1: Upload CSV/Excel File",
        "Step 2: Select Independent Variables",
        "Step 3: Choose Model & Assign Weights",
        "Step 4: Your Results",
        "Demo"
    ]
    st.sidebar.title("📖 Navigation")
    for i, t in enumerate(titles):
        if i == st.session_state.step:
            st.sidebar.markdown(f"### ✅ **{t}**")
        else:
            st.sidebar.markdown(f"### {t}")
    st.sidebar.markdown("---")
    st.sidebar.button("🔄 Restart", on_click=reset_app, key='restart')

# -------------------------------------------------------------------------------
# Model Functions
# -------------------------------------------------------------------------------

def run_linear_regression(X, y):
    st.subheader("📈 Linear Regression Results")
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(y.name, axis=1)
    y = data[y.name]
    X = add_constant(X)
    model = OLS(y, X).fit()
    preds = model.predict(X)
    st.write("**Regression Summary:**")
    st.text(model.summary())
    coef_df = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std. Error': model.bse.values,
        'P-Value': model.pvalues.values
    })
    st.write("**Coefficients:**")
    st.dataframe(coef_df)
    r2 = model.rsquared
    st.write(f"**R²:** {r2:.4f}")
    fig = px.scatter(
        x=y, y=preds,
        labels={'x':'Actual','y':'Predicted'},
        title=f'Actual vs. Predicted ({y.name})',
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)
    results_df = pd.DataFrame({'Actual': y, 'Predicted': preds, 'Residual': y-preds})
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", data=csv,
                       file_name='linear_regression_results.csv', mime='text/csv')

def run_lightgbm(X, y):
    st.subheader("⚡ LightGBM Regression Results")
    st.markdown("**⚠️ Note:** Training may take a few minutes.")
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    if X.shape[0] < 200:
        st.error("❌ Need at least 200 rows for LightGBM.")
        return
    test_size = st.slider("Test size %", 10, 50, 20, 5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )
    st.write(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    tune = st.checkbox("Hyperparameter Tuning", True)
    if tune:
        grid = {
            'num_leaves':[31,50], 'max_depth':[10,20],
            'learning_rate':[0.01,0.05],'n_estimators':[100,200],
            'subsample':[0.8,1.0],'colsample_bytree':[0.8,1.0]
        }
        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            lgbm, grid, n_iter=10, cv=3,
            scoring='neg_mean_squared_error',
            random_state=42, n_jobs=-1, verbose=1
        )
        with st.spinner("🔄 Tuning..."):
            try:
                search.fit(X_train, y_train)
                best = search.best_estimator_
                st.write(f"**Best Params:** {search.best_params_}")
            except Exception as e:
                st.error(f"❌ Tuning error: {e}")
                return
        scores = cross_val_score(
            best, X_train, y_train, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        st.write(f"CV MSE: {-scores.mean():.2f} ± {scores.std():.2f}")
    else:
        best = lgb.LGBMRegressor(
            num_leaves=31, max_depth=10, learning_rate=0.05,
            n_estimators=100, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        with st.spinner("🔄 Training..."):
            try:
                best.fit(X_train, y_train)
                st.success("Trained!")
            except Exception as e:
                st.error(f"❌ Training error: {e}")
                return
    try:
        preds = best.predict(X_test)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2s = r2_score(y_test, preds)
    st.write(f"MSE: {mse:.2f} | MAE: {mae:.2f} | R²: {r2s:.4f}")
    # Rank list
    st.markdown("### 📋 Rank-Ordered Adoption 2025")
    df_accounts = st.session_state.df
    if 'acct_numb' in df_accounts and 'acct_name' in df_accounts:
        idx = X_test.index
        acct = df_accounts.loc[idx, ['acct_numb','acct_name']].reset_index(drop=True)
        out = pd.DataFrame({
            'acct_numb':acct['acct_numb'],
            'acct_name':acct['acct_name'],
            'Predicted_Adoption_2025':preds
        })
    else:
        out = pd.DataFrame({
            'Account_Index':X_test.index,
            'Predicted_Adoption_2025':preds
        })
    out = out.sort_values(out.columns[-1], ascending=False).reset_index(drop=True)
    out['Rank'] = out.index+1
    def style_top(r):
        if r['Rank']==1: return ['background-color:gold']*len(r)
        if r['Rank']==2: return ['background-color:silver']*len(r)
        if r['Rank']==3: return ['background-color:#cd7f32']*len(r)
        return ['']*len(r)
    st.dataframe(out.style.apply(style_top, axis=1), use_container_width=True)
    # SHAP
    try:
        st.markdown("### 🧠 SHAP Explainability")
        expl = shap.TreeExplainer(best)
        sv = expl.shap_values(X_test)
        fig = plt.figure(figsize=(10,6))
        shap.summary_plot(sv, X_test, plot_type="bar", show=False)
        st.pyplot(fig, use_container_width=True)
        plt.clf()
    except Exception as e:
        st.warning(f"⚠️ SHAP error: {e}")
    csv_out = out.to_csv(index=False).encode('utf-8')
    st.download_button("Download Ranked CSV", data=csv_out,
                       file_name='lightgbm_ranked.csv', mime='text/csv')

def run_weighted_scoring_model(df, weights, target_column, mappings):
    st.subheader("⚖️ Weighted Scoring Results")
    df_enc = encode_categorical_features(df.copy(), mappings)
    df_enc['Weighted_Score'] = 0
    for feat, w in weights.items():
        if feat in df_enc and pd.api.types.is_numeric_dtype(df_enc[feat]):
            df_enc['Weighted_Score'] += df_enc[feat] * w
    df_enc['Weighted_Score'] = df_enc['Weighted_Score'].replace([np.inf,-np.inf],np.nan)
    before = df_enc.shape[0]
    df_enc = df_enc.dropna(subset=['Weighted_Score'])
    if df_enc.shape[0]<before:
        st.warning(f"Dropped {before-df_enc.shape[0]} invalid rows.")
    if df_enc.empty:
        st.error("❌ No valid scores to rank.")
        return
    df_enc['Rank'] = df_enc['Weighted_Score'].rank(method='dense',ascending=False).astype(int)
    df_enc['Adopter_Category'] = pd.qcut(
        df_enc['Rank'], q=3,
        labels=['Early Adopter','Middle Adopter','Late Adopter']
    )
    emojis = {'Early Adopter':'🚀','Middle Adopter':'⏳','Late Adopter':'🐢'}
    st.markdown("### 🏆 Leaderboard")
    top_n = st.slider("Top N accounts", 5, 50, 10, 1)
    cols = ['acct_numb','acct_name','Weighted_Score','Rank','Adopter_Category', target_column]
    if all(c in df_enc for c in ['acct_numb','acct_name']):
        top = df_enc[cols].nlargest(top_n,'Weighted_Score')
    else:
        top = df_enc[['Weighted_Score','Rank','Adopter_Category',target_column]] \
              .nlargest(top_n,'Weighted_Score') \
              .reset_index(drop=True)
        top.index += 1
    def style_cat(r):
        if r['Adopter_Category']=='Early Adopter': return ['background-color:lightgreen']*len(r)
        if r['Adopter_Category']=='Middle Adopter': return ['background-color:lightyellow']*len(r)
        if r['Adopter_Category']=='Late Adopter': return ['background-color:lightcoral']*len(r)
        return ['']*len(r)
    st.dataframe(top.style.apply(style_cat,axis=1), use_container_width=True)
    csv_scores = top.to_csv(index=False).encode('utf-8')
    st.download_button("Download Scores CSV", data=csv_scores,
                       file_name='weighted_scores.csv', mime='text/csv')

def run_demo():
    st.title("🎥 Demo")
    st.markdown("### Coming Soon!")

# Render sidebar and main logic

render_sidebar()

if st.session_state.step == 0:
    st.title("💊 Behavior Prediction Platform 💊")
    st.markdown("""
    1. Upload Your Data  
    2. Select Variables  
    3. Choose Model  
    4. View Results  
    """)

elif st.session_state.step == 1:
    st.title("💊 Behavior Prediction Platform 💊")
    st.subheader("Step 1: Upload Your CSV/Excel File")
    st.download_button(
        "Download CSV Template 📄",
        data=pd.DataFrame({
            'acct_numb':['123','456','789'],
            'acct_name':['A','B','C']
        }).to_csv(index=False),
        file_name='template.csv', mime='text/csv'
    )
    uploaded_file = st.file_uploader("Choose your CSV or Excel file:", type=["csv","xlsx"])
    if uploaded_file:
        try:
            file_type = uploaded_file.type
            # first pass
            if file_type.endswith("sheet"):
                excel = pd.ExcelFile(uploaded_file)
                sheets = excel.sheet_names
                df_pref = pd.read_excel(uploaded_file, sheet_name=sheets[0], header=None)
            else:
                df_pref = pd.read_csv(uploaded_file, header=None)
            st.markdown("### Preview:")
            st.dataframe(df_pref.head())
            # detect headers
            possible = []
            for i,row in df_pref.iterrows():
                text_ct = row.apply(lambda x: isinstance(x,str)).sum()
                if len(row) and text_ct/len(row)>0.7:
                    possible.append(i)
            if possible:
                if len(possible)==1:
                    hr = possible[0]
                    st.session_state.header_row = hr
                    st.success(f"Header row {hr} auto-detected")
                else:
                    hr = st.selectbox("Select header row:", possible)
                    st.session_state.header_row = hr
            else:
                hr = st.number_input("Enter header row index:",0,len(df_pref)-1,0)
                st.session_state.header_row = hr
            # **FIX** rewind
            uploaded_file.seek(0)
            if file_type.endswith("sheet"):
                df = pd.read_excel(
                    uploaded_file,
                    sheet_name= sheets[0] if len(sheets)==1 else st.session_state.sheet_selection,
                    header=st.session_state.header_row
                )
            else:
                df = pd.read_csv(uploaded_file, header=st.session_state.header_row)
            df = generate_account_adoption_rank(df)
            st.session_state.df = df
            st.success("File processed!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error: {e}")

elif st.session_state.step == 2:
    st.title("💊 Behavior Prediction Platform 💊")
    df = st.session_state.df
    if df is None:
        st.warning("Upload file first.")
    else:
        st.subheader("Step 2: Select Independent Variables")
        ids = ['acct_numb','acct_name']
        feats = [c for c in df.columns if c not in ids]
        sel = st.multiselect("Select features:", feats, default=feats[:3])
        if sel:
            st.session_state.selected_features = sel
            st.success(f"{len(sel)} features selected")

elif st.session_state.step == 3:
    st.title("💊 Behavior Prediction Platform 💊")
    df = st.session_state.df
    sel = st.session_state.selected_features
    if not sel:
        st.warning("Select features first.")
    else:
        st.subheader("Step 3: Choose Model & Assign Weights")
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button("Run Linear Regression"):
                select_linear_regression()
        with c2:
            if st.button("Run Weighted Scoring"):
                select_weighted_scoring()
        with c3:
            if st.button("Run LightGBM"):
                select_lightgbm()
        if st.session_state.selected_model in ['linear_regression','lightgbm']:
            st.info("Select your dependent variable:")
            # **FIX** use sel, not undefined set
            targets = [c for c in df.columns if c not in ['acct_numb','acct_name'] + sel]
            tgt = st.selectbox("Dependent variable:", options=targets)
            if tgt:
                st.session_state.target_column = tgt
                st.success(f"Target set to {tgt}")
        else:
            # weighted scoring UI...
            pass

elif st.session_state.step == 4:
    st.title("💊 Behavior Prediction Platform 💊")
    m = st.session_state.selected_model
    if m == 'linear_regression':
        run_linear_regression(st.session_state.X, st.session_state.y)
    elif m == 'lightgbm':
        run_lightgbm(st.session_state.X, st.session_state.y)
    elif m == 'weighted_scoring_model':
        run_weighted_scoring_model(
            st.session_state.df,
            st.session_state.normalized_weights,
            'Account Adoption Rank Order',
            categorical_mappings
        )
    c1,c2 = st.columns(2)
    with c1:
        st.button("← Back to Step 3", on_click=reset_to_step_3)
    with c2:
        st.button("🔄 Restart", on_click=reset_app)

elif st.session_state.step == 5:
    run_demo()

if st.session_state.step < 5:
    b1,b2 = st.columns([1,1])
    with b1:
        if st.session_state.step>0:
            st.button("← Back", on_click=prev_step)
    with b2:
        st.button("Next →", on_click=next_step)

# Bottom navigation
if st.session_state.step < 5:
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.session_state.step > 0:
            st.button("← Back", on_click=prev_step, key="back_bottom")
    with b2:
        st.button("Next →", on_click=next_step, key="next_bottom")# Bottom navigation
if st.session_state.step < 5:
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.session_state.step > 0:
            st.button("← Back", on_click=prev_step, key="back_bottom")
    with b2:
        st.button("Next →", on_click=next_step, key="next_bottom")
