# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize session state variables for navigation and selections
if 'step' not in st.session_state:
    st.session_state.step = 1  # Current step: 1 to 4

if 'df' not in st.session_state:
    st.session_state.df = None  # Uploaded DataFrame

if 'target_column' not in st.session_state:
    st.session_state.target_column = None  # Selected target variable

if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []  # Selected independent variables

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None  # Selected model

# Function to reset the app to Step 1
def reset_app():
    st.session_state.step = 1
    st.session_state.df = None
    st.session_state.target_column = None
    st.session_state.selected_features = []
    st.session_state.selected_model = None

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

def run_linear_regression(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    st.subheader("📈 Linear Regression Results")
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
    """
    Trains and evaluates a Random Forest Regressor.
    """
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
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, predictions, alpha=0.7)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Actual vs Predicted")
    st.pyplot(fig2)

def run_weighted_scoring_model(df, feature_weights, target_column, mappings):
    """
    Calculates and evaluates a Weighted Scoring Model based on selected features and their weights.
    """
    st.subheader("⚖️ Weighted Scoring Model Results")
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df.copy(), mappings)
    
    # Calculate weighted score
    df_encoded['Weighted_Score'] = 0
    for feature, weight in feature_weights.items():
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
    fig, ax = plt.subplots()
    ax.scatter(df_encoded['Weighted_Score'], df_encoded[target_column], alpha=0.7)
    ax.set_xlabel("Weighted Score")
    ax.set_ylabel(target_column)
    ax.set_title("Weighted Score vs Actual")
    st.pyplot(fig)

# Step 1: Upload CSV and Download Template
if st.session_state.step == 1:
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
            'Total_2022_and_2023': [50000, 60000, 54000],
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
    
    # Next button
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df  # Store in session state
            st.success("✅ File uploaded successfully!")
            st.session_state.step = 2  # Move to next step
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

# Step 2: Select Target Variable
elif st.session_state.step == 2:
    df = st.session_state.df
    st.subheader("Step 2: Select Target Variable")
    
    # Identify numeric columns for target selection
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_columns:
        st.error("❌ No numeric columns found in the uploaded data to select as target.")
        if st.button("Run a New Model 🔄"):
            reset_app()
    else:
        target_column = st.radio(
            "Choose one target variable for prediction:",
            options=numeric_columns
        )
        
        # Next and Back buttons
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Back"):
                st.session_state.step = 1
        with col2:
            if st.button("Next"):
                st.session_state.target_column = target_column
                st.session_state.step = 3

# Step 3: Select Independent Variables
elif st.session_state.step == 3:
    df = st.session_state.df
    target_column = st.session_state.target_column
    st.subheader("Step 3: Select Independent Variables")
    
    # Exclude target, 'acct_numb', and 'acct_name' from features
    possible_features = [col for col in df.columns if col not in [target_column, 'acct_numb', 'acct_name']]
    
    # Multiselect for feature selection
    selected_features = st.multiselect(
        "Choose your independent variables (features):",
        options=possible_features,
        default=possible_features[:3],  # Default selection
        help="Select one or more features to include in the model."
    )
    
    # Next and Back buttons
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Back"):
            st.session_state.step = 2
    with col2:
        if st.button("Next"):
            if selected_features:
                st.session_state.selected_features = selected_features
                st.session_state.step = 4
            else:
                st.warning("⚠️ Please select at least one independent variable.")

# Step 4: Choose and Run Model
elif st.session_state.step == 4:
    df = st.session_state.df
    target_column = st.session_state.target_column
    selected_features = st.session_state.selected_features
    
    st.subheader("Step 4: Choose and Run Model")
    st.write("Select the predictive model you want to run based on your selected features.")
    
    # Define feature weights for Weighted Scoring Model
    # Assign default weights (1.0) to each selected feature
    feature_weights = {feature: 1.0 for feature in selected_features}
    
    # Allow user to adjust weights if desired (optional)
    st.write("**Assign Weights to Selected Features (Optional):**")
    for feature in selected_features:
        weight = st.number_input(
            f"Weight for {feature}",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key=f"weight_{feature}"
        )
        feature_weights[feature] = weight
    
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
    
    # Model Selection Buttons with Descriptions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Run Linear Regression", key='lr'):
            st.session_state.selected_model = 'linear_regression'
    
        if hyperlink_button("Description", key='lr_desc'):
            st.session_state.show_description['linear_regression'] = not st.session_state.show_description['linear_regression']
    
        style_hyperlink_button('lr_desc')
    
        if st.session_state.show_description['linear_regression']:
            st.info("**Linear Regression:** Choose this model if you're working with between 10-50 lines of data.")
    
    with col2:
        if st.button("Run Random Forest", key='rf'):
            st.session_state.selected_model = 'random_forest'
    
        if hyperlink_button("Description", key='rf_desc'):
            st.session_state.show_description['random_forest'] = not st.session_state.show_description['random_forest']
    
        style_hyperlink_button('rf_desc')
    
        if st.session_state.show_description['random_forest']:
            st.info("**Random Forest:** Choose this model if you're working with >50 lines of data and want to leverage predictive power.")
    
    with col3:
        if st.button("Run Weighted Scoring Model", key='wsm'):
            st.session_state.selected_model = 'weighted_scoring_model'
    
        if hyperlink_button("Description", key='wsm_desc'):
            st.session_state.show_description['weighted_scoring_model'] = not st.session_state.show_description['weighted_scoring_model']
    
        style_hyperlink_button('wsm_desc')
    
        if st.session_state.show_description['weighted_scoring_model']:
            st.info("**Weighted Scoring Model:** Choose this model if you're looking for analysis, not prediction.")
    
    # Run Model Button
    if st.session_state.selected_model:
        if st.button("Run Model"):
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
            
            # Split the data into training and testing sets
            test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5, key='test_size_slider')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            
            st.write(f"**Training samples:** {X_train.shape[0]} | **Testing samples:** {X_test.shape[0]}")
            
            # Execute selected model with loading spinner
            if st.session_state.selected_model == 'linear_regression':
                with st.spinner("Training Linear Regression model..."):
                    run_linear_regression(X_train, X_test, y_train, y_test)
            elif st.session_state.selected_model == 'random_forest':
                with st.spinner("Training Random Forest model..."):
                    run_random_forest(X_train, X_test, y_train, y_test)
            elif st.session_state.selected_model == 'weighted_scoring_model':
                with st.spinner("Calculating Weighted Scoring Model..."):
                    run_weighted_scoring_model(df, feature_weights, target_column, categorical_mappings)
    
    # Back and Reset Buttons
    st.markdown("<hr>", unsafe_allow_html=True)
    col_back, col_reset = st.columns([1,1])
    with col_back:
        if st.button("Back"):
            st.session_state.step = 3
            st.session_state.selected_model = None
    with col_reset:
        if st.button("Run a New Model 🔄"):
            reset_app()
