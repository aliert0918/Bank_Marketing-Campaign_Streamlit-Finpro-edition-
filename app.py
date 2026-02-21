import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. KONFIGURASI HALAMAN & HEADER
# ==========================================
st.set_page_config(page_title="Telemarketing Term Deposit Prediction", layout="wide")

st.title("Bank Marketing Term Deposit Prediction")
st.markdown("### Machine Learning-Based Decision Support System for Telemarketing Effectiveness")

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
    # model.estimator_ digunakan jika model utama adalah wrapper seperti TunedThresholdClassifierCV
    pipeline_internal = model.estimator_ 
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# 3. INPUT FITUR (Dikelompokkan)
# ==========================================
st.markdown("---")
st.markdown("### 📋 Customer Information Input")

tab1, tab2, tab3, tab4 = st.tabs(["Client Information", "Last Contact Information", "Socio-Economic Context", "Engineered Features"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        # Range age 17-98 
        age = st.number_input("Age", min_value=17, max_value=98, value=40)
        # Unique categories from [cite: 1]
        job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
        education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'])
    with col2:
        default = st.selectbox("Has Credit in Default?", ['no', 'unknown', 'yes'])
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        contact = st.selectbox("Contact Type", ['telephone', 'cellular'])
        month = st.selectbox("Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])
    with col4:
        campaign = st.number_input("Contacts during Campaign", min_value=1, max_value=56, value=2)
        
        # LOGIKA PDAYS: 999 menunjukkan belum pernah dikontak 
        not_contacted_before = st.checkbox("Belum pernah dikontak sebelumnya?", value=True)
        if not_contacted_before:
            pdays = 999
            st.caption("Pdays otomatis diatur ke 999")
        else:
            pdays = st.number_input("Days since last contact", min_value=0, max_value=27, value=7)
            
        previous = st.number_input("Previous Contacts", min_value=0, max_value=7, value=0)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        # Socio-economic ranges from 
        emp_var_rate = st.slider("Employment Var Rate", -3.4, 1.4, 0.1)
        cons_price_idx = st.slider("Consumer Price Index", 92.201, 94.767, 93.575)
        cons_conf_idx = st.slider("Consumer Confidence Index", -50.8, -26.9, -40.5)
    with col6:
        euribor3m = st.slider("Euribor 3 Month Rate", 0.634, 5.045, 3.621)
        nr_employed = st.slider("Number of Employees", 4963.6, 5228.1, 5167.0)

with tab4:
    col7, col8 = st.columns(2)
    with col7:
        contacted_before = st.selectbox("Contacted Before (Binary)", [0, 1], index=0 if pdays == 999 else 1)
        previous_success = st.selectbox("Previous Success (Binary)", [0, 1], index=1 if poutcome == 'success' else 0)
    with col8:
        is_success_month = st.selectbox("Is Success Month (Binary)", [0, 1], index=0)
        euribor_low = st.selectbox("Euribor Low (Binary)", [0, 1], index=0)

# ==========================================
# 4. PREDIKSI & HASIL
# ==========================================
if st.button("Predict Term Deposit Conversion", type="primary"):
    # Buat dictionary sesuai urutan di column_names.txt 
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
    
    # Tampilkan Tabel Info Customer
    st.markdown("### 🧑‍💼 Customer Profile Summary")
    st.dataframe(df_input)
    
    # Prediksi
    prediction = model.predict(df_input)[0]
    
    # Ambil Probabilitas
    try:
        prob = model.predict_proba(df_input)[0][1]
    except AttributeError:
        prob = pipeline_internal.predict_proba(df_input)[0][1]

    # Display Hasil
    st.markdown("### 🎯 Prediction Result")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction == 1:
            st.success("Result: **YES** (Likely to subscribe)")
        else:
            st.error("Result: **NO** (Unlikely to subscribe)")
    with res_col2:
        st.metric("Conversion Probability", f"{prob * 100:.2f}%")
