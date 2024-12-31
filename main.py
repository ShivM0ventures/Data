import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import time

# Set page config and styling
st.set_page_config(
    page_title="Loan Repayment Prediction 💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7f9 0%, #e8f4f8 100%);
        padding: 2.5rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #2196F3, #00BCD4);
        color: white;
        padding: 0.8rem 2.5rem;
        border-radius: 25px;
        border: none;
        transition: all 0.3s;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #1976D2, #0097A7);
        box-shadow: 0 5px 15px rgba(33,150,243,0.3);
        transform: translateY(-2px);
    }
    .css-1v0mbdj.ebxwdo61 {
        border-radius: 15px;
        padding: 2rem;
        background: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(4px);
    }
    h1 {
        background: linear-gradient(45deg, #1565C0, #0097A7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 1rem;
        border-bottom: 3px solid #2196F3;
        font-size: 2.5rem !important;
        text-align: center;
    }
    h3 {
        color: #1565C0;
        margin-top: 2rem;
        font-weight: 600;
    }
    .stNumberInput, .stSlider {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    .stSlider > div > div {
        background: linear-gradient(90deg, #2196F3, #00BCD4);
    }
    .prediction-box {
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .feature-label {
        font-weight: 600;
        color: #1565C0;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background: rgba(33,150,243,0.1);
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved scaler and model
with st.spinner('Loading models... 🔄'):
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('loan_default_model_cb.pkl')
    st.success('Models loaded successfully! ✅')

# [Keep the feature_cols, numerical_cols, categorical_cols, and preprocess_input function as they are]
[...]

# Streamlit app
def main():
    st.title("Loan Repayment Prediction 🎯")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                padding: 1.5rem; 
                border-radius: 15px; 
                margin-bottom: 2rem;
                border-left: 5px solid #2196F3;'>
        <h4 style='color: #1565C0; margin: 0;'>
            🎉 Welcome! Enter your details below to get an instant loan repayment prediction.
        </h4>
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='feature-label'>📊 Basic Information</div>", unsafe_allow_html=True)
        age = st.number_input("Age 👤", min_value=18, max_value=100, value=30)
        cash_incoming_30days = st.number_input("Cash Incoming in Last 30 Days (KES) 💵", 
                                             min_value=0.0, value=5000.0,
                                             help="Total money received in the last month")

    with col2:
        st.markdown("<div class='feature-label'>📍 GPS-based Features</div>", unsafe_allow_html=True)
        gps_fix_count = st.number_input("Number of App Opens 📱", min_value=0, value=10)
        unique_locations_count = st.number_input("Unique Locations Visited 🗺️", min_value=0, value=5)

    col3, col4 = st.columns(2)

    with col3:
        avg_time_between_opens = st.number_input("Average Time Between App Opens (seconds) ⏱️", 
                                               min_value=0.0, value=3600.0)
        night_usage_ratio = st.slider("Nighttime Activity Ratio 🌙", 
                                    min_value=0.0, max_value=1.0, value=0.2,
                                    help="Proportion of app usage during night hours")

    with col4:
        num_clusters = st.number_input("Number of Significant Locations 📍", min_value=0, value=2)

    # [Keep the user_input dictionary as is]
    [...]

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Loan Outcome"):
        with st.spinner('Analyzing your data... 🤔'):
            # Add artificial delay for better UX
            time.sleep(1)
            
            input_data = preprocess_input(user_input)
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            outcome = 'Repaid ✅' if prediction[0] == 1 else 'Defaulted ❌'
            proba = prediction_proba[0][prediction[0]]

            # Celebration effect for positive outcome
            if prediction[0] == 1:
                st.balloons()

            result_color = '#1565C0' if outcome == 'Repaid ✅' else '#D32F2F'
            st.markdown(f"""
                <div class='prediction-box' style='
                    background: linear-gradient(135deg, {result_color}15, {result_color}25);
                    padding: 2rem;
                    border-radius: 20px;
                    margin-top: 2rem;
                    border: 2px solid {result_color}50;
                    text-align: center;'>
                    <h2 style='color: {result_color}; margin-bottom: 1rem;'>
                        {outcome}
                    </h2>
                    <div style='
                        background: white;
                        border-radius: 15px;
                        padding: 1rem;
                        display: inline-block;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <h3 style='color: {result_color}; margin: 0;'>
                            Confidence: {proba:.2%}
                        </h3>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Display input data in a collapsible section
            with st.expander("🔍 View Detailed Analysis"):
                st.markdown("<div class='info-box'>Here's how we analyzed your data:</div>", 
                          unsafe_allow_html=True)
                st.dataframe(input_data.style.background_gradient(cmap='Blues'))

if __name__ == '__main__':
    main()
