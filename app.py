import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lime
import lime.lime_tabular
import streamlit.components.v1 as components

# ==========================================
# 1. KONFIGURASI HALAMAN & HEADER
# ==========================================
st.set_page_config(page_title="Telemarketing Term Deposit Prediction", layout="wide")

st.title("Bank Marketing Term Deposit Prediction")
st.markdown("### Machine Learning-Based Decision Support System for Telemarketing Effectiveness")

# Dashboard Team Info
st.sidebar.markdown("### 👥 Team Information")
st.sidebar.info(
    "**JCDSBDGPM10+AM08 DELTA GROUP**\n\n"
    "- Alifsya Salam\n"
    "- Salma Almira Kuswihandono\n"
    "- Wahyu Eki Sepriansyah"
)

# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model_path = "threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    # Mengambil internal pipeline untuk LIME dan preprocessing
    pipeline_internal = model.estimator_ 
    prep = pipeline_internal.named_steps['preprocessing']
    xgb_mod = pipeline_internal.named_steps['modeling']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# 3. INPUT FITUR (Grouped Tabs)
# ==========================================
st.markdown("---")
st.markdown("### 📋 Customer Information Input")

tab1, tab2, tab3, tab4 = st.tabs(["Client Profile", "Contact Detail", "Socio-Economic", "Feature Engineering"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=17, max_value=98, value=40)
        job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    with col2:
        education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'])
        default = st.selectbox("Has Credit in Default?", ['no', 'unknown', 'yes'])
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        contact = st.selectbox("Contact Communication Type", ['telephone', 'cellular'])
        month = st.selectbox("Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    with col4:
        campaign = st.number_input("Contacts during Campaign", min_value=1, max_value=56, value=2)
        # Logika Pdays: checkbox untuk 999
        not_contacted_before = st.checkbox("Belum pernah dikontak sebelumnya (Pdays 999)", value=True)
        pdays = 999 if not_contacted_before else st.number_input("Days since last contact", min_value=0, max_value=27, value=7)
        previous = st.number_input("Previous Contacts", min_value=0, max_value=7, value=0)
        poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])

with tab3:
    # socio-economic ditampilkan langsung (tidak disembunyikan)
    st.subheader("Macroeconomic Indicators")
    col5, col6 = st.columns(2)
    with col5:
        emp_var_rate = st.slider("Employment Variation Rate", -3.4, 1.4, 0.08)
        cons_price_idx = st.slider("Consumer Price Index", 92.201, 94.767, 93.575, format="%.3f")
    with col6:
        cons_conf_idx = st.slider("Consumer Confidence Index", -50.8, -26.9, -40.5)
        euribor3m = st.slider("Euribor 3 Month Rate", 0.634, 5.045, 3.621, format="%.3f")
        nr_employed = st.slider("Number of Employees", 4963.6, 5228.1, 5167.0)

with tab4:
    col7, col8 = st.columns(2)
    with col7:
        contacted_before = st.selectbox("Contacted Before (1=Yes, 0=No)", [0, 1], index=0 if pdays == 999 else 1)
        previous_success = st.selectbox("Previous Success (1=Yes, 0=No)", [0, 1], index=1 if poutcome == 'success' else 0)
    with col8:
        is_success_month = st.selectbox("Is Success Month (1=Yes, 0=No)", [0, 1], index=0)
        euribor_low = st.selectbox("Euribor Low (1=Yes, 0=No)", [0, 1], index=0)

# ==========================================
# 4. PREDICTION & LIME
# ==========================================
if st.button("Predict Term Deposit Conversion", type="primary"):
    # Urutan kolom sesuai column_names.txt
    input_dict = {
        'age': age, 'job': job, 'marital': marital, 'education': education, 
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact, 
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign, 
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome, 
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed, 
        'contacted_before': contacted_before, 'previous_success': previous_success, 
        'is_success_month': is_success_month, 'euribor_low': euribor_low
    }
    
    df_input = pd.DataFrame([input_dict])
    
    st.markdown("### 🧑‍💼 Customer Profile Table")
    st.dataframe(df_input)
    
    # Predict
    prediction = model.predict(df_input)[0]
    try:
        prob = model.predict_proba(df_input)[0][1]
    except AttributeError:
        prob = pipeline_internal.predict_proba(df_input)[0][1]

    # Result Display
    st.markdown("### 🎯 Prediction Result")
    res1, res2 = st.columns(2)
    with res1:
        if prediction == 1:
            st.success("Recommendation: **SUBSCRIBE** (YES)")
        else:
            st.error("Recommendation: **DO NOT SUBSCRIBE** (NO)")
    with res2:
        st.metric("Conversion Probability", f"{prob * 100:.2f}%")

    # LIME Explanation
    st.markdown("---")
    st.markdown("### 🔍 Customer Analysis (LIME)")
    with st.spinner("Generating LIME analysis..."):
        try:
            # Transform input ke numerik
            transformed_input = prep.transform(df_input)
            
            # Buat background data simpel dari input
            dummy_data = pd.concat([df_input]*50, ignore_index=True)
            transformed_background = prep.transform(dummy_data)
            
            feature_names = prep.get_feature_names_out()

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=transformed_background,
                feature_names=feature_names,
                class_names=['No', 'Yes'],
                mode='classification'
            )
            
            # Predict function mengarah ke XGB modeling step
            exp = explainer.explain_instance(
                data_row=transformed_input[0],
                predict_fn=xgb_mod.predict_proba,
                num_features=10
            )
            
            components.html(exp.as_html(), height=450, scrolling=True)
        except Exception as e:
            st.warning(f"LIME Analysis unavailable: {e}")
