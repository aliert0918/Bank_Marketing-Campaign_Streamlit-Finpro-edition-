import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. LOAD MODEL & SETUP
# ==========================================
st.set_page_config(page_title="Bank Marketing Deployment", layout="wide")

MODEL_FILE = 'threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_FILE, 'rb') as file:
        return pickle.load(file)

model_wrapper = load_model()

# ==========================================
# 2. DEFINISI INPUT (Berdasarkan Dataset)
# ==========================================
# Data kategori dari kolom_kategori_unique_values.csv [cite: 2]
# Data numerik dari kolom_numerik_range.csv [cite: 3]

st.title("🏦 Bank Marketing Campaign Predictor")
st.info("Aplikasi ini memprediksi apakah nasabah akan berlangganan deposit berjangka.")

with st.form("main_form"):
    # Group 1: Demographics
    st.subheader("👤 Profil Nasabah")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Umur", 17, 98, 38) # [cite: 3]
        job = st.selectbox("Pekerjaan", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student']) # [cite: 2]
        marital = st.selectbox("Status Pernikahan", ['married', 'single', 'divorced', 'unknown']) # [cite: 2]
    with col2:
        education = st.selectbox("Pendidikan", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate']) # [cite: 2]
        default = st.selectbox("Memiliki Gagal Bayar?", ['no', 'unknown', 'yes']) # [cite: 2]
        housing = st.selectbox("Pinjaman Perumahan?", ['no', 'yes', 'unknown']) # [cite: 2]
        loan = st.selectbox("Pinjaman Pribadi?", ['no', 'yes', 'unknown']) # [cite: 2]

    # Group 2: Campaign History
    st.subheader("📞 Riwayat Kampanye")
    col3, col4 = st.columns(2)
    with col3:
        contact = st.selectbox("Alat Komunikasi", ['telephone', 'cellular']) # [cite: 2]
        month = st.selectbox("Bulan Kontak Terakhir", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep']) # [cite: 2]
        day_of_week = st.selectbox("Hari Kontak Terakhir", ['mon', 'tue', 'wed', 'thu', 'fri']) # [cite: 2]
        campaign = st.number_input("Jumlah Kontak Selama Kampanye Ini", 1, 56, 1) # [cite: 3]
    with col4:
        pdays = st.number_input("Hari Sejak Kontak Terakhir (999=Belum Pernah)", 0, 999, 999) # [cite: 3]
        previous = st.number_input("Jumlah Kontak Sebelum Kampanye Ini", 0, 7, 0) # [cite: 3]
        poutcome = st.selectbox("Hasil Kampanye Sebelumnya", ['nonexistent', 'failure', 'success']) # [cite: 2]

    # Group 3: Economic & Engineered Features
    st.subheader("📈 Indikator Ekonomi & Fitur Tambahan")
    col5, col6 = st.columns(2)
    with col5:
        emp_var_rate = st.number_input("Employment Variation Rate", -3.4, 1.4, 1.1) # [cite: 3]
        cons_price_idx = st.number_input("Consumer Price Index", 92.201, 94.767, 93.994) # [cite: 3]
        cons_conf_idx = st.number_input("Consumer Confidence Index", -50.8, -26.9, -36.4) # [cite: 3]
    with col6:
        euribor3m = st.number_input("Euribor 3 Month Rate", 0.634, 5.045, 4.857) # [cite: 3]
        nr_employed = st.number_input("Number of Employees", 4963.6, 5228.1, 5228.1) # [cite: 3]
        
        # Engineered Features dari column_names.txt 
        contacted_before = st.selectbox("Pernah Dikontak Sebelumnya? (1=Ya, 0=Tidak)", [0, 1])
        previous_success = st.selectbox("Sukses di Masa Lalu? (1=Ya, 0=Tidak)", [0, 1])
        is_success_month = st.selectbox("Bulan Sukses? (1=Ya, 0=Tidak)", [0, 1])
        euribor_low = st.selectbox("Euribor Rendah? (1=Ya, 0=Tidak)", [0, 1])

    submitted = st.form_submit_button("Prediksi")

# ==========================================
# 3. PREDICTION & SHAP LOGIC
# ==========================================
if submitted:
    # Buat DataFrame sesuai urutan column_names.txt 
    input_df = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed,
        'contacted_before': contacted_before, 'previous_success': previous_success,
        'is_success_month': is_success_month, 'euribor_low': euribor_low
    }])

    # Prediksi menggunakan wrapper (menggunakan threshold yang sudah di-tune)
    prediction = model_wrapper.predict(input_df)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.success("### HASIL: Nasabah diprediksi akan BERLANGGANAN (YES) ✅")
    else:
        st.error("### HASIL: Nasabah diprediksi TIDAK akan berlangganan (NO) ❌")

    # --- Bagian SHAP (Revisi untuk XGBoost) ---
    st.subheader("🔍 Penjelasan Model (SHAP)")
    try:
        # 1. Bongkar TunedThresholdClassifierCV untuk mendapatkan Pipeline
        # Biasanya pipeline ada di atribut .estimator_ (dengan underscore di akhir setelah fit)
        pipeline = model_wrapper.estimator_ if hasattr(model_wrapper, 'estimator_') else model_wrapper.estimator
        
        # 2. Ambil transformer dan model XGBoost
        # Sesuaikan 'transformer' dan 'model' dengan nama step di pipeline kamu
        # Jika kamu tidak yakin namanya, gunakan pipeline.named_steps.keys()
        transformer = pipeline.named_steps['transformer'] 
        xgb_model = pipeline.named_steps['model']

        # 3. Transform data input
        X_transformed = transformer.transform(input_df)
        
        # 4. Hitung SHAP
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        # 5. Visualisasi
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Ambil nama fitur setelah transformasi (untuk OneHotEncoding)
        feature_names = transformer.get_feature_names_out()
        
        # Plot force plot untuk data pertama (indeks 0)
        # Jika binary classification, shap_values bisa berupa list atau array 2D
        expected_value = explainer.expected_value
        current_shap = shap_values[0] if len(shap_values.shape) == 2 else shap_values[0][:, 1]

        fig = shap.force_plot(
            expected_value, 
            current_shap, 
            X_transformed[0], 
            feature_names=feature_names, 
            matplotlib=True, 
            show=False
        )
        st.pyplot(fig, bbox_inches='tight')
        
    except Exception as e:
        st.warning(f"Gagal memuat SHAP: {e}")
        st.info("Tips: Pastikan nama step di pipeline adalah 'transformer' dan 'model'.")
