import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
from imblearn.pipeline import Pipeline # Important for loading the pickle

# ==========================================
# 1. LOAD MODEL & HELPERS
# ==========================================
st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")

MODEL_FILE = 'threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_FILE, 'rb') as file:
        return pickle.load(file)

# We load the 'wrapper' (TunedThresholdClassifierCV)
model_wrapper = load_model()

# Function to automate feature engineering (so the user doesn't have to)
def apply_feature_engineering(df):
    df = df.copy()
    df["contacted_before"] = (df["pdays"] != 999).astype(int)
    df["previous_success"] = (df["poutcome"] == "success").astype(int)
    success_months = ["mar", "dec", "sep", "oct"]
    df["is_success_month"] = df["month"].isin(success_months).astype(int)
    df["euribor_low"] = (df["euribor3m"] <= 1.5).astype(int)
    return df

# ==========================================
# 2. UI LAYOUT (Grouped Inputs)
# ==========================================
st.title("🏦 Bank Marketing Subscription Predictor")

with st.form("prediction_form"):
    # Group 1: Client Profile
    st.subheader("👤 Client Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 17, 98, 30)
        job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])
    with c2:
        marital = st.selectbox("Marital", ['married', 'single', 'divorced', 'unknown'])
        education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'])
    with c3:
        default = st.selectbox("Default?", ['no', 'unknown', 'yes'])
        housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])

    st.divider()

    # Group 2: Campaign Data
    st.subheader("📞 Campaign Details")
    c4, c5, c6 = st.columns(3)
    with c4:
        contact = st.selectbox("Contact Type", ['telephone', 'cellular'])
        month = st.selectbox("Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])
    with c5:
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.number_input("Contacts in this campaign", 1, 50, 1)
    with c6:
        pdays = st.number_input("Days since last contact (999=Never)", 0, 999, 999)
        previous = st.number_input("Previous contacts", 0, 10, 0)
        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])

    st.divider()

    # Group 3: Economic Indicators
    st.subheader("📊 Socio-Economic Indicators")
    c7, c8, c9 = st.columns(3)
    with c7:
        emp_var_rate = st.number_input("Emp. Var. Rate", -4.0, 2.0, 1.1)
        cons_price_idx = st.number_input("Cons. Price Index", 90.0, 95.0, 93.9)
    with c8:
        cons_conf_idx = st.number_input("Cons. Conf. Index", -50.0, -10.0, -36.4)
        euribor3m = st.number_input("Euribor 3M", 0.0, 6.0, 4.8)
    with c9:
        nr_employed = st.number_input("No. Employed", 4900.0, 5300.0, 5228.0)

    predict_btn = st.form_submit_button("Predict & Explain")

# ==========================================
# 3. PREDICTION & SHAP FIX
# ==========================================
if predict_btn:
    # 1. Prepare Base DataFrame
    raw_input = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }])

    # 2. Auto-Engineer the 4 extra features
    final_input_df = apply_feature_engineering(raw_input)

    # 3. Predict using the Wrapper (uses tuned threshold)
    pred = model_wrapper.predict(final_input_df)[0]
    
    if pred == 1:
        st.success("### Prediction: YES! Client likely to subscribe. ✅")
    else:
        st.error("### Prediction: NO. Client unlikely to subscribe. ❌")

    # 4. SHAP EXPLANATION
    st.subheader("🔍 Why did the model say that?")
    try:
        # STEP A: Unwrap the Pipeline from the TunedThresholdClassifierCV
        # Based on your file, it's stored in .estimator_
        pipeline = model_wrapper.estimator_ 
        
        # STEP B: Identify specific steps
        # From your metadata: Step 0 is 'preprocessing', Step 1 is 'model'
        preprocessor = pipeline.named_steps['preprocessing']
        xgb_model = pipeline.named_steps['modeling']

        # STEP C: Transform data for SHAP
        # SHAP needs the numeric/encoded data that goes INTO the XGBoost
        X_transformed = preprocessor.transform(final_input_df)
        
        # STEP D: Calculate SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        # STEP E: Visualizing (Force Plot)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Get feature names from preprocessor
        feature_names = preprocessor.get_feature_names_out()

        # Handle SHAP output format (XGBoost can return log-odds)
        if isinstance(shap_values, list): # For some versions/multi-class
            sv = shap_values[1]
            ev = explainer.expected_value[1]
        else:
            sv = shap_values
            ev = explainer.expected_value

        fig = shap.force_plot(
            ev, 
            sv[0], 
            X_transformed[0], 
            feature_names=feature_names, 
            matplotlib=True, 
            show=False
        )
        st.pyplot(fig, bbox_inches='tight')
        st.caption("Red features push the prediction toward YES, Blue features toward NO.")

    except Exception as e:
        st.error(f"SHAP Error: {e}")
        st.info("Check if your pipeline step names are exactly 'preprocessing' and 'modeling'.")
