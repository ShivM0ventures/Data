import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier

# Set page config and styling
st.set_page_config(
    page_title="Loan Repayment Prediction",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .css-1v0mbdj.ebxwdo61 {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        padding-bottom: 1rem;
        border-bottom: 2px solid #4CAF50;
    }
    h3 {
        color: #34495e;
        margin-top: 2rem;
    }
    .stNumberInput, .stSlider {
        background-color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('loan_default_model_cb.pkl')

# Define the feature columns
feature_cols = [
    'age',
    'log_cash_incoming_30days',
    'gps_fix_count',
    'unique_locations_count',
    'avg_time_between_opens',
    'night_usage_ratio',
    'num_clusters',
    'income_bracket_Medium',
    'income_bracket_High',
    'income_bracket_Very High'
]

# Define numerical and categorical columns
numerical_cols = [
    'age',
    'log_cash_incoming_30days',
    'gps_fix_count',
    'unique_locations_count',
    'avg_time_between_opens',
    'night_usage_ratio',
    'num_clusters'
]

categorical_cols = [
    'income_bracket_Medium',
    'income_bracket_High',
    'income_bracket_Very High'
]

# Function to preprocess user input
def preprocess_input(user_input):
    # Create DataFrame
    input_df = pd.DataFrame([user_input])

    # Handle missing GPS features if any
    gps_feature_cols = [
        'gps_fix_count',
        'unique_locations_count',
        'avg_time_between_opens',
        'night_usage_ratio',
        'num_clusters'
    ]
    input_df[gps_feature_cols] = input_df[gps_feature_cols].fillna(0)

    # Log transformation for cash_incoming_30days
    input_df['log_cash_incoming_30days'] = np.log1p(input_df['cash_incoming_30days'])

    # Income brackets
    cash_incoming = input_df['cash_incoming_30days'].values[0]
    if cash_incoming < 2000:
        income_bracket = 'Low'
    elif cash_incoming < 5000:
        income_bracket = 'Medium'
    elif cash_incoming < 10000:
        income_bracket = 'High'
    else:
        income_bracket = 'Very High'

    # One-hot encoding for income bracket
    for bracket in ['Medium', 'High', 'Very High']:
        col_name = f'income_bracket_{bracket}'
        input_df[col_name] = 1 if income_bracket == bracket else 0

    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Select the columns in the correct order
    input_df = input_df[feature_cols]

    return input_df

# Streamlit app
def main():
    st.title("Loan Repayment Prediction")

    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 1rem; border-radius: 5px; margin-bottom: 2rem;'>
        Enter your details to check the loan repayment prediction.
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # User inputs
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        cash_incoming_30days = st.number_input("Cash Incoming in Last 30 Days (KES)", min_value=0.0, value=5000.0)

    with col2:
        st.markdown("### GPS-based Features (optional)")
        gps_fix_count = st.number_input("Number of App Opens (GPS Fix Count)", min_value=0, value=10)
        unique_locations_count = st.number_input("Unique Locations Visited", min_value=0, value=5)

    col3, col4 = st.columns(2)

    with col3:
        avg_time_between_opens = st.number_input("Average Time Between App Opens (seconds)", min_value=0.0, value=3600.0)
        night_usage_ratio = st.slider("Nighttime Activity Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.2)

    with col4:
        num_clusters = st.number_input("Number of Significant Locations (Clusters)", min_value=0, value=2)

    # Prepare user input
    user_input = {
        'age': age,
        'cash_incoming_30days': cash_incoming_30days,
        'gps_fix_count': gps_fix_count,
        'unique_locations_count': unique_locations_count,
        'avg_time_between_opens': avg_time_between_opens,
        'night_usage_ratio': night_usage_ratio,
        'num_clusters': num_clusters
    }

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Loan Outcome"):
        # Add a spinner during prediction
        with st.spinner('Processing...'):
            # Preprocess input
            input_data = preprocess_input(user_input)

            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # Display result
            outcome = 'Repaid' if prediction[0] == 1 else 'Defaulted'
            proba = prediction_proba[0][prediction[0]]

            # Style the prediction output
            result_color = '#4CAF50' if outcome == 'Repaid' else '#f44336'
            st.markdown(f"""
                <div style='background-color: {result_color}22; padding: 2rem; border-radius: 10px; margin-top: 2rem;'>
                    <h2 style='color: {result_color}; margin-bottom: 1rem;'>Prediction: {outcome}</h2>
                    <h3 style='color: {result_color}99;'>Probability: {proba:.2f}</h3>
                </div>
            """, unsafe_allow_html=True)

            # Display input data in a collapsible section
            with st.expander("View Input Data"):
                st.dataframe(input_data.style.background_gradient(cmap='Blues'))

if __name__ == '__main__':
    main()
