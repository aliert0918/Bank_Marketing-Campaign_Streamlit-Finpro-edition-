import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
from imblearn.pipeline import Pipeline

# ==========================================
# 1. LOAD MODEL
# ==========================================
st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")

MODEL_FILE = 'threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_FILE, 'rb') as file:
        return pickle.load(file)

# Load model wrapper (TunedThresholdClassifierCV)
model = load_model()

def add_engineered_features(df):
    df = df.copy()
    df["contacted_before"] = (df["pdays"] != 999).astype(int)
    df["previous_success"] = (df["poutcome"] == "success").astype(int)
    success_months = ["mar", "dec", "sep", "oct"]
    df["is_success_month"] = df["month"].isin(success_months).astype(int)
    df["euribor_low"] = (df["euribor3m"] <= 1.5).astype(int)
    return df

# ==========================================
# 2. UI INPUT
# ==========================================
st.title("🏦 Bank Marketing Prediction Tool")

with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 17, 98, 30)
        job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])
        marital = st.selectbox("Marital", ['married', 'single', 'divorced', 'unknown'])
    with col2:
        education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'])
        default = st.selectbox("Default?", ['no', 'unknown', 'yes'])
        housing = st.selectbox("Housing?", ['no', 'yes', 'unknown'])
    with col3:
        loan = st.selectbox("Loan?", ['no', 'yes', 'unknown'])
        contact = st.selectbox("Contact", ['telephone', 'cellular'])
        month = st.selectbox("Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])

    col4, col5, col6 = st.columns(3)
    with col4:
        day_of_week = st.selectbox("Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.number_input("Campaign", 1, 50, 1)
        pdays = st.number_input("Pdays", 0, 999, 999)
    with col5:
        previous = st.number_input("Previous", 0, 10, 0)
        poutcome = st.selectbox("Poutcome", ['nonexistent', 'failure', 'success'])
        emp_var_rate = st.number_input("Emp.Var.Rate", -4.0, 2.0, 1.1)
    with col6:
        cons_price_idx = st.number_input("Cons.Price.Idx", 90.0, 95.0, 93.9)
        cons_conf_idx = st.number_input("Cons.Conf.Idx", -50.0, -10.0, -36.4)
        euribor3m = st.number_input("Euribor3m", 0.0, 6.0, 4.8)
        nr_employed = st.number_input("Nr.Employed", 4900.0, 5300.0, 5228.0)

    predict_btn = st.form_submit_button("Predict & Explain")

# ==========================================
# 3. PREDICTION & SHAP
# ==========================================
if predict_btn:
    # 1. Prepare Data
    raw_input = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }])

    df_final = add_engineered_features(raw_input)

    # 2. Prediction
    prediction = model.predict(df_final)[0]
    
    st.divider()
    if prediction == 1:
        st.success("### Prediction: YES (Subscribe) ✅")
    else:
        st.error("### Prediction: NO (Will not subscribe) ❌")

    # 3. SHAP Explanation
    st.subheader("🔍 Kenapa Model Memprediksi Demikian?")
    try:
        # Bongkar wrapper ke pipeline internal
        pipeline_internal = model.estimator_ 
        
        # Ambil step 'preprocessing' dan 'modeling'
        prep = pipeline_internal.named_steps['preprocessing']
        xgb_mod = pipeline_internal.named_steps['modeling']
        
        # Transform data input
        X_transformed = prep.transform(df_final)
        
        # Kalkulasi SHAP
        explainer = shap.TreeExplainer(xgb_mod)
        shap_values = explainer.shap_values(X_transformed)
        
        # Dapatkan nama kolom fitur
        try:
            feat_names = prep.get_feature_names_out()
        except:
            feat_names = [f"Col_{i}" for i in range(X_transformed.shape[1])]

        # Plotting menggunakan cara yang aman (matplotlib figure)
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            X_transformed[0], 
            feature_names=feat_names,
            matplotlib=True, 
            show=False
        )
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.clf() # Bersihkan plot setelah tampil
        
        st.info("Merah: Mendorong ke arah YES | Biru: Mendorong ke arah NO")

    except Exception as e:
        st.error(f"Gagal memuat SHAP: {e}")
