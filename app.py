import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD MODEL & SETUP
# ==========================================
st.set_page_config(page_title="Bank Marketing Deployment", layout="wide")

MODEL_FILE = "threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl"

@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

model_wrapper = load_model(MODEL_FILE)

# ==========================================
# 2. FEATURE ENGINEERING (sesuai notebook vertopal)
# ==========================================
def feature_engineering(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    # Sama persis seperti notebook
    df["contacted_before"] = (df["pdays"] != 999).astype(int)
    df["previous_success"] = (df["poutcome"] == "success").astype(int)

    success_months = ["mar", "dec", "sep", "oct"]
    df["is_success_month"] = df["month"].isin(success_months).astype(int)

    df["euribor_low"] = (df["euribor3m"] <= 1.5).astype(int)

    return df

# ==========================================
# 3. UI INPUT
# ==========================================
st.title("Bank Marketing Campaign Predictor")
st.info("Aplikasi ini memprediksi apakah nasabah akan berlangganan deposit berjangka.")

with st.form("main_form"):
    st.subheader("Profil Nasabah")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Umur", 17, 98, 38)
        job = st.selectbox(
            "Pekerjaan",
            ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired',
             'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student']
        )
        marital = st.selectbox("Status Pernikahan", ['married', 'single', 'divorced', 'unknown'])

    with col2:
        education = st.selectbox(
            "Pendidikan",
            ['basic.4y', 'high.school', 'basic.6y', 'basic.9y',
             'professional.course', 'unknown', 'university.degree', 'illiterate']
        )
        default = st.selectbox("Memiliki Gagal Bayar?", ['no', 'unknown', 'yes'])
        housing = st.selectbox("Pinjaman Perumahan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Pinjaman Pribadi?", ['no', 'yes', 'unknown'])

    st.subheader("Riwayat Kampanye")
    col3, col4 = st.columns(2)

    with col3:
        contact = st.selectbox("Alat Komunikasi", ['telephone', 'cellular'])
        month = st.selectbox("Bulan Kontak Terakhir", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])
        day_of_week = st.selectbox("Hari Kontak Terakhir", ['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.number_input("Jumlah Kontak Selama Kampanye Ini", min_value=1, max_value=56, value=1)

    with col4:
        pdays = st.number_input("Hari Sejak Kontak Terakhir (999=Belum Pernah)", min_value=0, max_value=999, value=999)
        previous = st.number_input("Jumlah Kontak Sebelum Kampanye Ini", min_value=0, max_value=7, value=0)
        poutcome = st.selectbox("Hasil Kampanye Sebelumnya", ['nonexistent', 'failure', 'success'])

    st.subheader("Indikator Ekonomi")
    col5, col6 = st.columns(2)

    with col5:
        emp_var_rate = st.number_input("Employment Variation Rate", min_value=-3.4, max_value=1.4, value=1.1)
        cons_price_idx = st.number_input("Consumer Price Index", min_value=92.201, max_value=94.767, value=93.994)
        cons_conf_idx = st.number_input("Consumer Confidence Index", min_value=-50.8, max_value=-26.9, value=-36.4)

    with col6:
        euribor3m = st.number_input("Euribor 3 Month Rate", min_value=0.634, max_value=5.045, value=4.857)
        nr_employed = st.number_input("Number of Employees", min_value=4963.6, max_value=5228.1, value=5228.1)

    submitted = st.form_submit_button("Prediksi")

# ==========================================
# 4. PREDICTION & SHAP
# ==========================================
if submitted:
    # Input RAW sesuai fitur asli dataset (tanpa fitur turunan manual)
    input_raw = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }])

    # Terapkan feature engineering sesuai training
    input_df = feature_engineering(input_raw)

    # Prediksi label (sudah pakai tuned threshold di wrapper)
    pred = int(model_wrapper.predict(input_df)[0])

    # Probabilitas (kalau tersedia)
    proba = None
    if hasattr(model_wrapper, "predict_proba"):
        try:
            proba = float(model_wrapper.predict_proba(input_df)[0, 1])
        except Exception:
            proba = None

    st.markdown("---")
    if pred == 1:
        st.success("### HASIL: Nasabah diprediksi akan BERLANGGANAN (YES)")
    else:
        st.error("### HASIL: Nasabah diprediksi TIDAK akan berlangganan (NO)")

    if proba is not None:
        st.write(f"Probabilitas (kelas YES): **{proba:.3f}**")

    # --- SHAP untuk XGBoost ---
    st.subheader("Penjelasan Model (SHAP)")

    def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 400, scrolling=True)

# ... (Kode pipeline loading Anda sebelumnya) ...

# --- Bagian try-except SHAP yang baru ---
try:
    # 1. Identifikasi Preprocessor & Model (Gunakan index agar aman)
    preprocessor = model.steps[0][1]
    model_estimator = model.steps[-1][1]
    
    # 2. Transformasi data input
    X_processed = preprocessor.transform(input_df)
    
    # 3. Hitung SHAP Values
    explainer = shap.TreeExplainer(model_estimator)
    shap_values = explainer.shap_values(X_processed)
    
    # Handling format output SHAP (jika list, ambil kelas positif/index 1)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values_to_plot = shap_values
        expected_value = explainer.expected_value

    # 4. Ambil nama fitur (supaya grafik ada labelnya, bukan cuma Feature 0, 1...)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"Feature {i}" for i in range(X_processed.shape[1])]

    # 5. Visualisasi (HAPUS st.set_option yang bikin error)
    st.write("Grafik ini menunjukkan fitur mana yang mendorong prediksi ke arah 'YES' (merah) atau 'NO' (biru).")
    
    # Gunakan matplotlib=False agar grafiknya interaktif (bisa di-hover mouse)
    force_plot = shap.force_plot(
        expected_value,
        shap_values_to_plot[0], # Ambil data pertama
        X_processed[0],         # Ambil data pertama
        feature_names=feature_names,
        matplotlib=False        # PENTING: Set False agar jadi JavaScript
    )
    
    # Tampilkan menggunakan fungsi helper
    st_shap(force_plot)

except Exception as e:
    st.error(f"Gagal memuat SHAP: {e}")
