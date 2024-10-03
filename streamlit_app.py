# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to clear Streamlit cache
def clear_cache():
    try:
        st.cache_data.clear()
    except:
        pass
    try:
        st.cache_resource.clear()
    except:
        pass

# Clear cache at the beginning
clear_cache()

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

# Function to advance to the next step
def next_step():
    if st.session_state.step < 4:
        st.session_state.step += 1

# Function to go back to the previous step
def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1

# Function to reset the app to Step 1 and clear cache
def reset_app():
    st.session_state.step = 1
    st.session_state.df = None
    st.session_state.target_column = 'Account Adoption Rank Order'
    st.session_state.selected_features = []
    st.session_state.selected_model = None
    clear_cache()
    st.experimental_rerun()

# Define mappings for categorical features
categorical_mappings = {
    'analog_1_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_2_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_3_adopt': {'low': 1, 'medium': 2, 'high': 3}
}

# Helper function to encode categorical features
def encode_categorical_features(df, mappings):
    for feature, mapping in mappings.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping)
            if df[feature].isnull().any():
                st.warning(f"Some values in '{feature}' couldn't be mapped and are set to NaN.")
                df[feature].fillna(df[feature].mode()[0], inplace=True)
    return df

# Helper function to generate 'Account Adoption Rank Order'
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
    
    # Calculate total sales
    df['Total_Sales'] = df[sales_columns].sum(axis=1)
    
    # Generate rank order (1 being highest sales)
    df['Account Adoption Rank Order'] = df['Total_Sales'].rank(method='dense', ascending=False).astype(int)
    
    return df

# Model functions
def run_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return predictions, mse

def run_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    return predictions, mse, feature_importances

def run_weighted_scoring_model(df_encoded, normalized_weights, target_column):
    # Calculate weighted score
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
    
    return df_encoded, correlation

# Sidebar rendering
def render_sidebar():
    step_titles = [
        "Upload CSV File",
        "Confirm Target Variable",
        "Select Independent Variables",
        "Assign Weights & Choose Model"
    ]
    current_step = st.session_state.step

    st.sidebar.title("📖 Instructions")

    for i, title in enumerate(step_titles, 1):
        if i < current_step:
            st.sidebar.markdown(f"### ✅ Step {i}: {title}")
        elif i == current_step:
            st.sidebar.markdown(f"### **🔵 Step {i}: {title}**")
        else:
            st.sidebar.markdown(f"### Step {i}: {title}")

# Render the sidebar
render_sidebar()

# Step 1: Upload CSV and Download Template
if st.session_state.step == 1:
    st.title("💊 Pathways Prediction Platform")
    st.subheader("Step 1: Upload Your CSV File")
    
    # Download button for CSV template
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
            
            # Generate 'Account Adoption Rank Order'
            df = generate_account_adoption_rank(df)
            st.session_state.df = df  # Update in session state
            
            st.success("✅ File uploaded and 'Account Adoption Rank Order' generated successfully!")
            
            # Next button
            st.button("Next →", on_click=next_step, key='next_step1')
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

# Step 2: Confirm Target Variable
elif st.session_state.step == 2:
    st.subheader("Step 2: Confirm Target Variable")
    
    # Display the selected target variable
    target_column = st.session_state.target_column if st.session_state.target_column else 'Account Adoption Rank Order'
    st.session_state.target_column = target_column  # Ensure it's set
    
    st.markdown(f"**Selected Target Variable:** `{target_column}`")
    
    # Descriptive text
    st.write("""
        In the next step, you will select the independent variables that will be used to predict the **Account Adoption Rank Order**.
        This guided process ensures that you choose the most relevant features for accurate predictions.
    """)
    
    # Navigation buttons
    col1, col2 = st.columns([1,1])
    with col1:
        st.button("← Back", on_click=prev_step, key='back_step2')
    with col2:
        st.button("Next →", on_click=next_step, key='next_step2')

# Step 3: Select Independent Variables
elif st.session_state.step == 3:
    df = st.session_state.df
    target_column = st.session_state.target_column
    st.subheader("Step 3: Select Independent Variables")
    
    # Exclude target and identifier columns
    identifier_columns = ['acct_numb', 'acct_name']
    possible_features = [col for col in df.columns if col not in [target_column] + identifier_columns]
    
    # Feature selection
    selected_features = st.multiselect(
        "Choose your independent variables (features):",
        options=possible_features,
        default=possible_features[:3],  # Default selection
        help="Select one or more features to include in the model.",
        key='feature_selection'
    )
    
    # Navigation buttons
    col1, col2 = st.columns([1,1])
    with col1:
        st.button("← Back", on_click=prev_step, key='back_step3')
    with col2:
        if st.button("Next →", on_click=next_step, key='next_step3'):
            if selected_features:
                st.session_state.selected_features = selected_features
            else:
                st.warning("⚠️ Please select at least one independent variable.")

# Step 4: Assign Weights & Choose Model
elif st.session_state.step == 4:
    df = st.session_state.df
    target_column = st.session_state.target_column
    selected_features = st.session_state.selected_features
    
    st.subheader("Step 4: Assign Weights & Choose Model")
    st.write("Select the predictive model you want to run based on your selected features.")
    
    # Instructions
    st.markdown("**Assign Weights to Selected Features** 🎯")
    st.write("""
        Assign how important each feature is in determining the **Account Adoption Rank Order**. 
        The weights must add up to **10**. Use the sliders below to assign weights in multiples of **5**.
    """)
    
    # Initialize a dictionary to store user-assigned weights
    feature_weights = {}
    
    st.markdown("### 🔢 **Enter Weights for Each Feature (Multiples of 5):**")
    
    # Create sliders for each feature with step=5
    for feature in selected_features:
        weight = st.slider(
            f"Weight for **{feature}**:",
            min_value=0,
            max_value=10,
            step=5,
            value=0,  # Default initial value of 0
            key=f"weight_slider_{feature}"
        )
        feature_weights[feature] = weight
    
    # Calculate total weight
    total_weight = sum(feature_weights.values())
    
    # Display total weight with validation
    st.markdown("---")
    st.markdown("### 🎯 **Total Weight Assigned:**")
    
    if total_weight < 10:
        status = f"❗ Total weight is **{total_weight}**, which is less than **10**."
        color = "#FFC107"  # Yellow
    elif total_weight > 10:
        status = f"❗ Total weight is **{total_weight}**, which is more than **10**."
        color = "#DC3545"  # Red
    else:
        status = f"✅ Total weight is **{total_weight}**, which meets the requirement."
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
    
    st.markdown("---")
    
    # Model Selection Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Linear Regression", key='model_linear_regression'):
            st.session_state.selected_model = 'linear_regression'
    with col2:
        if st.button("Random Forest", key='model_random_forest'):
            st.session_state.selected_model = 'random_forest'
    with col3:
        if st.button("Weighted Scoring Model", key='model_weighted_scoring'):
            st.session_state.selected_model = 'weighted_scoring_model'
    
    # Show description based on selected model
    if st.session_state.selected_model:
        if st.session_state.selected_model == 'linear_regression':
            st.info("**Linear Regression:** Suitable for predicting continuous values based on linear relationships.")
        elif st.session_state.selected_model == 'random_forest':
            st.info("**Random Forest:** Ideal for handling complex datasets with non-linear relationships and interactions.")
        elif st.session_state.selected_model == 'weighted_scoring_model':
            st.info("**Weighted Scoring Model:** Choose this model if you're looking for analysis, not prediction.")
    
    # Run Model Button
    if st.session_state.selected_model:
        if st.button("Run Model", key='run_model'):
            # Preprocess data
            X = df[selected_features]
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
            
            # Execute selected model with loading spinner
            if st.session_state.selected_model == 'linear_regression':
                with st.spinner("Running Linear Regression..."):
                    predictions, mse = run_linear_regression(X, y)
                st.subheader("📈 Linear Regression Results")
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                
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
            
            elif st.session_state.selected_model == 'random_forest':
                # Split the data into training and testing sets
                test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5, key='test_size_slider_rf')
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
                
                st.write(f"**Training samples:** {X_train.shape[0]} | **Testing samples:** {X_test.shape[0]}")
                
                with st.spinner("Training Random Forest model..."):
                    predictions, mse, feature_importances = run_random_forest(X_train, X_test, y_train, y_test)
                st.subheader("🌲 Random Forest Results")
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                
                # Plot Feature Importance
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
            
            elif st.session_state.selected_model == 'weighted_scoring_model':
                # Encode categorical features
                df_encoded = encode_categorical_features(df.copy(), categorical_mappings)
                
                with st.spinner("Calculating Weighted Scoring Model..."):
                    df_encoded, correlation = run_weighted_scoring_model(df_encoded, normalized_weights, target_column)
                
                st.subheader("⚖️ Weighted Scoring Model Results")
                st.write(f"**Correlation between Weighted Score and {target_column}:** {correlation:.2f}")
                
                # Allow user to select number of top accounts to display
                top_n = st.slider("Select number of top accounts to display", min_value=5, max_value=20, value=10, step=1)
                top_accounts = df_encoded[['acct_numb', 'acct_name', 'Weighted_Score', target_column]].sort_values(by='Weighted_Score', ascending=False).head(top_n)
                st.write(f"**Top {top_n} Accounts Based on Weighted Score:**")
                st.dataframe(top_accounts)
                
                # Plot Weighted Score vs Actual
                fig = px.scatter(
                    df_encoded,
                    x='Weighted_Score',
                    y=target_column,
                    labels={'Weighted_Score': 'Weighted Score', target_column: target_column},
                    title='Weighted Score vs Actual'
                )
                st.plotly_chart(fig)
    
    st.markdown("---")
    
    # Navigation buttons
    col_back, col_run, col_reset = st.columns([1,1,1])
    with col_back:
        st.button("← Back", on_click=prev_step, key='back_step4')
    with col_run:
        pass  # Placeholder for alignment
    with col_reset:
        st.button("Run a New Model 🔄", on_click=reset_app, key='reset_app')
