import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
from imblearn.pipeline import Pipeline # Wajib ada jika pakai imblearn pipeline

# ==========================================
# 1. KONFIGURASI & LOAD MODEL
# ==========================================
st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")

MODEL_FILE = 'threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_FILE, 'rb') as file:
        return pickle.load(file)

model_wrapper = load_model()

# Fungsi untuk membuat 4 fitur tambahan otomatis
def add_engineered_features(df):
    df = df.copy()
    df["contacted_before"] = (df["pdays"] != 999).astype(int)
    df["previous_success"] = (df["poutcome"] == "success").astype(int)
    success_months = ["mar", "dec", "sep", "oct"]
    df["is_success_month"] = df["month"].isin(success_months).astype(int)
    df["euribor_low"] = (df["euribor3m"] <= 1.5).astype(int)
    return df

# ==========================================
# 2. UI INPUT USER
# ==========================================
st.title("🏦 Bank Marketing Prediction Tool")
st.markdown("Gunakan form di bawah untuk memprediksi apakah nasabah akan berlangganan deposit.")

with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 17, 98, 30)
        job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
        education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'])
        
    with col2:
        default = st.selectbox("Has Credit in Default?", ['no', 'unknown', 'yes'])
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])
        contact = st.selectbox("Contact Type", ['telephone', 'cellular'])
        month = st.selectbox("Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])

    with col3:
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.number_input("Contacts during Campaign", 1, 50, 1)
        pdays = st.number_input("Days since last contact (999=Never)", 0, 999, 999)
        previous = st.number_input("Previous Contacts", 0, 10, 0)
        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])

    st.subheader("Socio-Economic Indicators")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        emp_var_rate = st.number_input("Employment Var Rate", -4.0, 2.0, 1.1)
    with ec2:
        cons_price_idx = st.number_input("Cons Price Index", 90.0, 95.0, 93.9)
        cons_conf_idx = st.number_input("Cons Conf Index", -50.0, -10.0, -36.4)
    with ec3:
        euribor3m = st.number_input("Euribor 3 Month", 0.0, 6.0, 4.8)
        nr_employed = st.number_input("No. Employed", 4900.0, 5300.0, 5228.0)

    predict_btn = st.form_submit_button("Predict & Explain")

# ==========================================
# 3. LOGIKA PREDIKSI & SHAP
# ==========================================
if predict_btn:
    # Buat DataFrame awal (19 kolom)
    raw_input = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }])

    # Tambahkan fitur engineering (Total jadi 23 kolom sesuai model)
    df_final = add_engineered_features(raw_input)

    # Prediksi
    prediction = model_wrapper.predict(df_final)[0]
    
    st.divider()
    if prediction == 1:
        st.success("### HASIL: Nasabah Diprediksi BERLANGGANAN (YES) ✅")
    else:
        st.error("### HASIL: Nasabah Diprediksi TIDAK Berlangganan (NO) ❌")

    # BAGIAN SHAP (Kunci Keberhasilan)
    st.subheader("🔍 Penjelasan Faktor Prediksi (SHAP)")
    try:
        with st.spinner('Membongkar model dan menghitung SHAP...'):
            # Ambil pipeline dari dalam wrapper TunedThreshold
            pipeline = model_wrapper.estimator_ 
            
            # Ambil step sesuai nama yang kamu debug: 'preprocessing' & 'modeling'
            prep = pipeline.named_steps['preprocessing']
            model_xgb = pipeline.named_steps['modeling']
            
            # Transform data input ke format angka
            X_transformed = prep.transform(df_final)
            
            # Explainer khusus XGBoost
            explainer = shap.TreeExplainer(model_xgb)
            shap_values = explainer.shap_values(X_transformed)
            
            # Ambil nama fitur hasil transformasi (OneHot encoding dll)
            try:
                feature_names = prep.get_feature_names_out()
            except:
                feature_names = [f"Col_{i}" for i in range(X_transformed.shape[1])]

            # Plotting Force Plot
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            # Handle format output (expected value dan shap value untuk baris ke-0)
            ev = explainer.expected_value
            sv = shap_values[0]

            fig = shap.force_plot(
                ev, sv, X_transformed[0], 
                feature_names=feature_names, 
                matplotlib=True, show=False
            )
            st.pyplot(fig, bbox_inches='tight')
            st.info("Penjelasan: Merah mendorong ke arah 'YES', Biru mendorong ke arah 'NO'.")
            
    except Exception as e:
        st.warning(f"SHAP gagal: {e}")
            st.info("Saran: Cek apakah data input sudah lengkap dan sesuai format.")
