import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lime
import lime.lime_tabular
import streamlit.components.v1 as components

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="DELTA - Term Deposit Prediction", layout="wide")
st.title("Bank Marketing Term Deposit Prediction")
st.sidebar.markdown("### 👥 Team DELTA")
st.sidebar.info("- Alifsya Salam\n- Salma Almira\n- Wahyu Eki")

# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model_path = "threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl"
    with open(model_path, 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
    pipeline_internal = model.estimator_ 
    prep = pipeline_internal.named_steps['preprocessing']
    xgb_mod = pipeline_internal.named_steps['modeling']
except Exception as e:
    st.error(f"Gagal muat model: {e}")
    st.stop()

# ==========================================
# 3. INPUT USER (Disederhanakan untuk contoh)
# ==========================================
st.markdown("### 📋 Customer Information")
tab1, tab2, tab3 = st.tabs(["Profile", "Contact", "Socio-Economic"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 17, 98, 40)
        job = st.selectbox("Job", ['admin.','blue-collar','technician','services','management','retired','entrepreneur','self-employed','housemaid','unemployed','student','unknown'])
        marital = st.selectbox("Marital", ['married','single','divorced','unknown'])
    with col2:
        education = st.selectbox("Education", ['university.degree','high.school','basic.9y','professional.course','basic.4y','basic.6y','unknown','illiterate'])
        default = st.selectbox("Default?", ['no','unknown','yes'])
        housing = st.selectbox("Housing?", ['no','yes','unknown'])
        loan = st.selectbox("Loan?", ['no','yes','unknown'])

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        contact = st.selectbox("Contact", ['cellular','telephone'])
        month = st.selectbox("Month", ['may','jul','aug','jun','nov','apr','oct','sep','mar','dec'])
        day_of_week = st.selectbox("Day", ['mon','tue','wed','thu','fri'])
    with col4:
        campaign = st.number_input("Campaign Contacts", 1, 50, 1)
        not_contacted = st.checkbox("Belum pernah dikontak?", value=True)
        pdays = 999 if not_contacted else st.number_input("Pdays", 0, 27, 7)
        previous = st.number_input("Previous", 0, 7, 0)
        poutcome = st.selectbox("Poutcome", ['nonexistent','failure','success'])

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        emp_var_rate = st.slider("Emp.Var.Rate", -3.4, 1.4, 1.1)
        cons_price_idx = st.slider("Cons.Price.Idx", 92.2, 94.7, 93.9)
    with col6:
        cons_conf_idx = st.slider("Cons.Conf.Idx", -50.8, -26.9, -36.4)
        euribor3m = st.slider("Euribor3m", 0.6, 5.0, 4.8)
        nr_employed = st.slider("Nr.Employed", 4963.0, 5228.0, 5228.0)

# Feature Engineering Otomatis
contacted_before = 0 if pdays == 999 else 1
previous_success = 1 if poutcome == 'success' else 0
is_success_month = 1 if month in ['mar','sep','oct','dec'] else 0
euribor_low = 1 if euribor3m < 1.0 else 0

# ==========================================
# 4. PREDIKSI & FIXED LIME
# ==========================================
if st.button("Predict & Analyze", type="primary"):
    data = {
        'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
        'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m, 'nr.employed': nr_employed, 'contacted_before': contacted_before,
        'previous_success': previous_success, 'is_success_month': is_success_month, 'euribor_low': euribor_low
    }
    df_input = pd.DataFrame([data])
    
    # 1. Prediksi
    prob = model.predict_proba(df_input)[0][1]
    
    st.markdown("### 🎯 Prediction Results")
    st.metric("Probability of Success (YES)", f"{prob*100:.2f}%")

    # 2. LIME Visualization (FIXED)
    st.markdown("---")
    st.markdown("### 🔍 Why did the model say this?")
    
    with st.spinner("Calculating Feature Importance..."):
        # Transform input
        X_instance = prep.transform(df_input)
        
        # PENTING: Membuat data background yang bervariasi agar LIME bisa bikin BARPLOT
        # Kita buat 500 baris data acak di sekitar input agar ada perbandingan
        X_background = np.repeat(X_instance, 500, axis=0)
        noise = np.random.normal(0, 0.1, X_background.shape) 
        X_background_noisy = X_background + noise 

        # Ambil nama fitur yang bersih
        raw_names = prep.get_feature_names_out()
        clean_names = [n.split('__')[-1].replace('binary_enc_','') for n in raw_names]

        # Inisialisasi Explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_background_noisy, # Pakai data bervariasi
            feature_names=clean_names,
            class_names=['No', 'Yes'],
            mode='classification'
        )

        # Penjelasan
        exp = explainer.explain_instance(
            data_row=X_instance[0],
            predict_fn=xgb_mod.predict_proba, # Langsung ke XGBoost step
            num_features=10
        )

        # RENDER SEBAGAI HTML (WAJIB agar muncul Grafik Bar)
        html_content = exp.as_html()
        components.html(html_content, height=800, scrolling=True)
