# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set the title of the app
st.title("💊 Pathways Prediction Platform")
st.write("Upload your data, select target and features, and run predictive models.")

# Initialize session state for selected model and descriptions
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None

if 'show_description' not in st.session_state:
    st.session_state['show_description'] = {
        'linear_regression': False,
        'random_forest': False,
        'weighted_scoring_model': False
    }

# Step 1: Upload CSV and download template
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

# Add a little space for better UI design
st.markdown("<br>", unsafe_allow_html=True)

# Upload box for CSV
uploaded_file = st.file_uploader(
    "Now, choose your CSV file:", type="csv", label_visibility="visible"
)

# Define mappings for categorical features
categorical_mappings = {
    'analog_1_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_2_adopt': {'low': 1, 'medium': 2, 'high': 3},
    'analog_3_adopt': {'low': 1, 'medium': 2, 'high': 3}
}

# Define helper functions

def encode_categorical_features(df, mappings):
    for feature, mapping in mappings.items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping)
            if df[feature].isnull().any():
                st.warning(f"Some values in '{feature}' couldn't be mapped and are set to NaN.")
                df[feature].fillna(df[feature].mode()[0], inplace=True)
    return df

def run_linear_regression(X_train, X_test, y_train, y_test):
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
    st.subheader("⚖️ Weighted Scoring Model Results")
    
    # Encode categorical features
    df = encode_categorical_features(df, mappings)
    
    # Calculate weighted score
    score = 0
    for feature, weight in feature_weights.items():
        if feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                score += df[feature] * weight
            else:
                st.warning(f"Feature '{feature}' is not numeric and will be skipped.")
        else:
            st.warning(f"Feature '{feature}' not found in the data. Skipping its weight.")
    
    df['Weighted_Score'] = score
    
    # Correlation with target
    correlation = df['Weighted_Score'].corr(df[target_column])
    st.write(f"**Correlation between Weighted Score and {target_column}:** {correlation:.2f}")
    
    # Display top accounts based on score
    top_n = st.slider("Select number of top accounts to display", min_value=5, max_value=20, value=10, step=1)
    top_accounts = df[['acct_numb', 'acct_name', 'Weighted_Score', target_column]].sort_values(by='Weighted_Score', ascending=False).head(top_n)
    st.dataframe(top_accounts)
    
    # Plot Weighted Score vs Target
    fig, ax = plt.subplots()
    ax.scatter(df['Weighted_Score'], df[target_column], alpha=0.7)
    ax.set_xlabel("Weighted Score")
    ax.set_ylabel(target_column)
    ax.set_title("Weighted Score vs Actual")
    st.pyplot(fig)

# Step 2: Model Selection (only if the file has been uploaded)
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        # Identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Let user select the target column using radio buttons
        st.subheader("Step 2: Select Target and Features")
        st.write("**Select the Target Variable:**")
        target_column = st.radio(
            "Choose one target variable for prediction:",
            options=numeric_columns
        )
        
        # List of possible features (excluding target, acct_numb, acct_name)
        possible_features = [col for col in numeric_columns + categorical_columns if col not in [target_column, 'acct_numb', 'acct_name']]
        
        # Let user select the number of independent variables
        max_features = len(possible_features)
        st.write("**Select Number of Independent Variables:**")
        num_features = st.slider(
            "How many independent variables do you want to include?",
            min_value=1,
            max_value=min(10, max_features),
            value=3,
            step=1
        )
        
        # Let user select specific features
        st.write("**Select Independent Variables:**")
        selected_features = st.multiselect(
            "Choose your features:",
            options=possible_features,
            default=possible_features[:num_features],
            help="Select the features you want to include in the model."
        )
        
        # Ensure the number of selected features matches the user's choice
        if len(selected_features) < num_features:
            st.warning(f"You have selected fewer features ({len(selected_features)}) than specified ({num_features}).")
        elif len(selected_features) > num_features:
            st.warning(f"You have selected more features ({len(selected_features)}) than specified ({num_features}). Please adjust your selection.")
        
        # Proceed only if the number of selected features matches
        if len(selected_features) == num_features:
            # Preprocess data based on selected features
            X = df[selected_features]
            y = df[target_column]
            
            # Handle categorical variables using one-hot encoding
            selected_categorical = [col for col in selected_features if col in categorical_columns]
            if selected_categorical:
                X = pd.get_dummies(X, columns=selected_categorical, drop_first=True)
            
            # Handle missing values (simple strategy: fill with mean for numeric, mode for categorical)
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col].fillna(X[col].mode()[0], inplace=True)
                else:
                    X[col].fillna(X[col].mean(), inplace=True)
            
            # Split the data into training and testing sets
            test_size = st.slider("Select Test Size Percentage", min_value=10, max_value=50, value=20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            
            st.write(f"**Training samples:** {X_train.shape[0]} | **Testing samples:** {X_test.shape[0]}")
            
            # Define feature weights based on selected features
            # For simplicity, assign a default weight of 1.0 to each selected feature
            feature_weights = {feature: 1.0 for feature in selected_features}
            # If you have predefined weights, you can update this dictionary accordingly
            # Example:
            # predefined_weights = {'feature1': 1.2, 'feature2': 1.1, ...}
            # feature_weights = {feature: predefined_weights.get(feature, 1.0) for feature in selected_features}
            
            # Define model selection and descriptions
            st.subheader("Step 3: Choose a Model")
            
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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Run Linear Regression", key='lr'):
                    st.session_state['selected_model'] = 'linear_regression'
            
                if hyperlink_button("Description", key='lr_desc'):
                    st.session_state['show_description']['linear_regression'] = not st.session_state['show_description']['linear_regression']
            
                style_hyperlink_button('lr_desc')
            
                if st.session_state['show_description']['linear_regression']:
                    st.info("**Linear Regression:** Choose this model if you're working with between 10-50 lines of data.")
            
            with col2:
                if st.button("Run Random Forest", key='rf'):
                    st.session_state['selected_model'] = 'random_forest'
            
                if hyperlink_button("Description", key='rf_desc'):
                    st.session_state['show_description']['random_forest'] = not st.session_state['show_description']['random_forest']
            
                style_hyperlink_button('rf_desc')
            
                if st.session_state['show_description']['random_forest']:
                    st.info("**Random Forest:** Choose this model if you're working with >50 lines of data and want to leverage predictive power.")
            
            with col3:
                if st.button("Run Weighted Scoring Model", key='wsm'):
                    st.session_state['selected_model'] = 'weighted_scoring_model'
            
                if hyperlink_button("Description", key='wsm_desc'):
                    st.session_state['show_description']['weighted_scoring_model'] = not st.session_state['show_description']['weighted_scoring_model']
            
                style_hyperlink_button('wsm_desc')
            
                if st.session_state['show_description']['weighted_scoring_model']:
                    st.info("**Weighted Scoring Model:** Choose this model if you're looking for analysis, not prediction.")
            
            # Execute selected model with loading spinner
            if st.session_state['selected_model'] == 'linear_regression':
                with st.spinner("Training Linear Regression model..."):
                    run_linear_regression(X_train, X_test, y_train, y_test)
            elif st.session_state['selected_model'] == 'random_forest':
                with st.spinner("Training Random Forest model..."):
                    run_random_forest(X_train, X_test, y_train, y_test)
            elif st.session_state['selected_model'] == 'weighted_scoring_model':
                with st.spinner("Calculating Weighted Scoring Model..."):
                    run_weighted_scoring_model(df, feature_weights, target_column, categorical_mappings)
            
            # Add a "Run a New Model" button after the model has run
            if st.session_state['selected_model'] is not None:
                st.markdown("<hr>", unsafe_allow_html=True)
                if st.button("Run a New Model 🔄"):
                    # Reset the session state
                    st.session_state['selected_model'] = None
                    st.session_state['show_description'] = {
                        'linear_regression': False,
                        'random_forest': False,
                        'weighted_scoring_model': False
                    }
                    st.experimental_rerun()
        else:
            st.warning("Please select the exact number of features specified using the slider.")
    
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
