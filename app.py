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

# Dashboard Team Info di Sidebar
st.sidebar.markdown("### 👥 Team Information")
st.sidebar.info(
    "**JCDSBDGPM10+AM08 DELTA GROUP**\n\n"
    "- Alifsya Salam\n"
    "- Salma Almira Kuswihandono\n"
    "- Wahyu Eki Sepriansyah"
)

# ==========================================
# 2. LOAD MODEL & PIPELINE STEPS
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan nama file pkl sesuai dengan yang ada di folder Anda
    model_path = "threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    # Mengekstrak pipeline internal dari wrapper TunedThresholdClassifierCV
    pipeline_internal = model.estimator_ 
    # Mengambil step preprocessing dan modeling (XGBoost)
    prep = pipeline_internal.named_steps['preprocessing']
    xgb_mod = pipeline_internal.named_steps['modeling']
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
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
        # Logika Pdays: checkbox untuk otomatis 999
        not_contacted_before = st.checkbox("Belum pernah dikontak sebelumnya (Pdays 999)", value=True)
        pdays = 999 if not_contacted_before else st.number_input("Days since last contact", min_value=0, max_value=27, value=7)
        previous = st.number_input("Previous Contacts", min_value=0, max_value=7, value=0)
        poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])

with tab3:
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
# 4. PREDICTION & FIXED LIME VISUALIZATION
# ==========================================
if st.button("Predict Term Deposit Conversion", type="primary"):
    # Menyiapkan data input sesuai urutan kolom asli model
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
    
    st.markdown("### 🧑‍💼 Customer Profile Summary")
    st.dataframe(df_input)
    
    # Proses Prediksi
    prediction = model.predict(df_input)[0]
    try:
        prob = model.predict_proba(df_input)[0][1]
    except AttributeError:
        prob = pipeline_internal.predict_proba(df_input)[0][1]

    # Menampilkan Hasil Prediksi
    st.markdown("### 🎯 Prediction Result")
    res1, res2 = st.columns(2)
    with res1:
        if prediction == 1:
            st.success("Recommendation: **SUBSCRIBE** (YES)")
        else:
            st.error("Recommendation: **DO NOT SUBSCRIBE** (NO)")
    with res2:
        st.metric("Conversion Probability", f"{prob * 100:.2f}%")

    # ==========================================
    # INTEGRASI FIX LIME (VISUAL & CLEAN NAMES)
    # ==========================================
    st.markdown("---")
    st.markdown("### 🔍 Customer Analysis (LIME)")
    st.info("Grafik di bawah menunjukkan fitur mana yang paling mempengaruhi keputusan model (Biru: Mendukung YES, Merah: Mendukung NO).")
    
    with st.spinner("Generating LIME analysis..."):
        try:
            # 1. Transformasi data input ke numerik melalui pipeline preprocessing
            transformed_input = prep.transform(df_input)
            
            # 2. Membuat background data untuk referensi LIME (menggunakan replikasi input)
            dummy_data = pd.concat([df_input]*100, ignore_index=True)
            transformed_background = prep.transform(dummy_data)
            
            # 3. PROSES MEMBERSIHKAN NAMA FITUR (Fix: No more 'onehot__' prefixes)
            try:
                raw_feature_names = prep.get_feature_names_out()
                clean_feature_names = [
                    name.split('__')[-1] # Mengambil nama asli setelah prefix '__'
                    .replace('binary_enc_', '') # Menghapus prefix binary encoder jika ada
                    for name in raw_feature_names
                ]
            except AttributeError:
                # Fallback jika get_feature_names_out tidak tersedia
                clean_feature_names = [f"Feature_{i}" for i in range(transformed_input.shape[1])]

            # 4. Inisiasi LIME Explainer dengan nama fitur yang sudah bersih
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=transformed_background,
                feature_names=clean_feature_names,
                class_names=['No (Gagal)', 'Yes (Sukses)'],
                mode='classification',
                random_state=42
            )
            
            # 5. Generate penjelasan LIME menggunakan modeling step (XGBoost)
            exp = explainer.explain_instance(
                data_row=transformed_input[0],
                predict_fn=xgb_mod.predict_proba,
                num_features=10
            )
            
            # 6. FIX VISUALISASI: Tampilkan sebagai HTML komponen (Grafik Bar)
            html_content = exp.as_html()
            components.html(html_content, height=500, scrolling=True)
            
        except Exception as e:
            st.warning(f"LIME Analysis unavailable: {e}")
            st.write("Saran: Pastikan library 'lime' sudah terinstal di environment Anda.")
