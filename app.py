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

    # Bagian Penjelasan Model (SHAP)
st.subheader("Penjelasan Model (SHAP)")

try:
    # 1. Identifikasi Preprocessor dan Model secara Dinamis
    # Alih-alih menebak nama ('transformer' vs 'preprocessor'), kita ambil berdasarkan urutan.
    # Biasanya: langkah pertama [0] adalah preprocessor, langkah terakhir [-1] adalah model.
    
    preprocessor = model.steps[0][1]  # Mengambil object transformer
    model_estimator = model.steps[-1][1] # Mengambil object model prediksi (misal: RandomForest/XGBoost)
    
    # 2. Transformasi data input
    # SHAP membutuhkan data yang sudah di-encode/scale, bukan data mentah string.
    X_processed = preprocessor.transform(input_df)
    
    # 3. Hitung SHAP Values
    # Menggunakan TreeExplainer (umum untuk Random Forest/Gradient Boosting)
    explainer = shap.TreeExplainer(model_estimator)
    shap_values = explainer.shap_values(X_processed)
    
    # Menangani format output SHAP (terkadang list, terkadang array tergantung model)
    if isinstance(shap_values, list):
        # Untuk klasifikasi biner, biasanya shap_values[1] adalah kelas positif (Yes/Berlangganan)
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # 4. Tampilkan Force Plot
    st.write("Grafik ini menunjukkan fitur mana yang mendorong prediksi ke arah 'YES' (merah) atau 'NO' (biru).")
    
    # Mendapatkan nama fitur setelah one-hot encoding (jika ada)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback jika model versi lama atau tidak support get_feature_names_out
        feature_names = [f"Feature {i}" for i in range(X_processed.shape[1])]

    # Visualisasi
    st_shap(shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
        shap_values_to_plot, 
        X_processed,
        feature_names=feature_names
    ))

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat SHAP: {e}")
    # Debugging: Tampilkan nama langkah yang tersedia di pipeline untuk pengecekan
    if hasattr(model, 'named_steps'):
        st.info(f"Nama langkah (steps) yang tersedia di model: {list(model.named_steps.keys())}")
