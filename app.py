import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ==========================================
# 1. PAGE CONFIGURATION & HEADER
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
    # Extract the internal pipeline from TunedThresholdClassifierCV
    pipeline_internal = model.estimator_ 
    prep = pipeline_internal.named_steps['preprocessing']
    xgb_mod = pipeline_internal.named_steps['modeling']
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# 3. FEATURE INPUTS
# ==========================================
st.markdown("---")
st.markdown("### 📋 Customer Information Input")

# We organize inputs into logical groups using tabs
tab1, tab2, tab3, tab4 = st.tabs(["Client Information", "Last Contact Information", "Socio-Economic Context", "Engineered Features"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=17, max_value=98, value=40, step=1)
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
        contact = st.selectbox("Contact Communication Type", ['telephone', 'cellular'])
        month = st.selectbox("Last Contact Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])
        day_of_week = st.selectbox("Last Contact Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        poutcome = st.selectbox("Outcome of Previous Campaign", ['nonexistent', 'failure', 'success'])
    with col4:
        campaign = st.number_input("Number of Contacts (Campaign)", min_value=1, max_value=56, value=2, step=1)
        pdays = st.number_input("Days Since Last Contact (999 means not previously contacted)", min_value=0, max_value=999, value=999, step=1)
        previous = st.number_input("Number of Contacts Before Campaign", min_value=0, max_value=7, value=0, step=1)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        emp_var_rate = st.slider("Employment Variation Rate", min_value=-3.4, max_value=1.4, value=0.08, step=0.1)
        cons_price_idx = st.slider("Consumer Price Index", min_value=92.201, max_value=94.767, value=93.575, step=0.001)
        cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-50.8, max_value=-26.9, value=-40.5, step=0.1)
    with col6:
        euribor3m = st.slider("Euribor 3 Month Rate", min_value=0.634, max_value=5.045, value=3.621, step=0.001)
        nr_employed = st.slider("Number of Employees", min_value=4963.6, max_value=5228.1, value=5167.0, step=0.1)

with tab4:
    st.info("These are derived features from your original dataset. Adjust them accordingly based on your logic.")
    col7, col8 = st.columns(2)
    with col7:
        contacted_before = st.selectbox("Contacted Before (1=Yes, 0=No)", [0, 1], index=0 if pdays == 999 else 1)
        previous_success = st.selectbox("Previous Success (1=Yes, 0=No)", [0, 1], index=1 if poutcome == 'success' else 0)
    with col8:
        is_success_month = st.selectbox("Is Success Month (1=Yes, 0=No)", [0, 1], index=0)
        euribor_low = st.selectbox("Euribor Low (1=Yes, 0=No)", [0, 1], index=0)

# ==========================================
# 4. PREDICTION & LIME EXPLANATION
# ==========================================
if st.button("Predict Term Deposit Conversion", type="primary"):
    # Create DataFrame
    input_data = {
        'age': age, 'job': job, 'marital': marital, 'education': education, 
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact, 
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign, 
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome, 
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed, 
        'contacted_before': contacted_before, 'previous_success': previous_success, 
        'is_success_month': is_success_month, 'euribor_low': euribor_low
    }
    
    # Ordered specifically as per column_names.txt
    column_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'contacted_before', 'previous_success', 'is_success_month', 'euribor_low']
    df_input = pd.DataFrame([input_data], columns=column_names)
    
    # 4.1 Display Customer Info Table
    st.markdown("### 🧑‍💼 Customer Profile Summary")
    st.dataframe(df_input)
    
    # 4.2 Prediction
    # Assuming the threshold model has predict and predict_proba methods wrapped cleanly
    prediction = model.predict(df_input)[0]
    
    # Some threshold tuners wrap predict_proba. Using the internal pipeline as fallback if needed:
    try:
        pred_proba = model.predict_proba(df_input)[0]
        prob_yes = pred_proba[1]
    except AttributeError:
        # Fallback to pipeline if TunedThresholdClassifierCV doesn't expose predict_proba directly
        pred_proba = pipeline_internal.predict_proba(df_input)[0]
        prob_yes = pred_proba[1]

    # 4.3 Result Display
    st.markdown("### 🎯 Prediction Result")
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        if prediction == 1:
            st.success("Result: YES (Customer is likely to subscribe to a term deposit)")
        else:
            st.error("Result: NO (Customer is unlikely to subscribe to a term deposit)")
    with result_col2:
        st.info(f"Conversion Probability: **{prob_yes * 100:.2f}%**")

    # 4.4 LIME Explanation
    st.markdown("### 🔍 Model Explanation (LIME)")
    with st.spinner("Generating explanation..."):
        try:
            # We explain on the TRANSFORMED data to bypass complex categorical mapping issues in Streamlit
            transformed_input = prep.transform(df_input)
            
            # Create a small dummy background dataset by transforming a synthetic raw df 
            # (In production, replace this with actual training data transformed)
            dummy_df = pd.concat([df_input]*10, ignore_index=True)
            transformed_background = prep.transform(dummy_df)
            
            # Fetch feature names outputted by the column transformer (if available)
            try:
                feature_names = prep.get_feature_names_out()
            except AttributeError:
                feature_names = [f"Feature_{i}" for i in range(transformed_background.shape[1])]

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=transformed_background,
                feature_names=feature_names,
                class_names=['No', 'Yes'],
                mode='classification'
            )
            
            def predict_fn(X):
                return xgb_mod.predict_proba(X)
                
            # Explain the instance
            exp = explainer.explain_instance(
                data_row=transformed_input[0],
                predict_fn=predict_fn,
                num_features=10
            )
            
            # Render LIME plot as HTML
            components.html(exp.as_html(), height=400, scrolling=True)
            
        except Exception as e:
            st.warning(f"Could not generate LIME explanation. Error: {str(e)}\n\n*Note: LIME often requires a representative background dataset during instantiation. Ensure your preprocessing step successfully outputs a dense array.*")
