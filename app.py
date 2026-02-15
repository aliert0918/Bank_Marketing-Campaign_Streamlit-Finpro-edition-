import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & LOAD DATA
# ==========================================
st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")

MODEL_FILE = 'threshold_tuned_BankMarketingFinpro_FOR_DEPLOYMENT_20260215_11_49.pkl'

# Manual definition of valid values based on provided source files [cite: 2, 3]
CATEGORICAL_OPTIONS = {
    'job': ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'],
    'marital': ['married', 'single', 'divorced', 'unknown'],
    'education': ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'],
    'default': ['no', 'unknown', 'yes'],
    'housing': ['no', 'yes', 'unknown'],
    'loan': ['no', 'yes', 'unknown'],
    'contact': ['telephone', 'cellular'],
    'month': ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'],
    'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
    'poutcome': ['nonexistent', 'failure', 'success']
}

NUMERICAL_RANGES = {
    'age': (17, 98, 38),
    'campaign': (1, 56, 2),
    'pdays': (0, 999, 999),
    'previous': (0, 7, 0),
    'emp.var.rate': (-3.4, 1.4, 1.1),
    'cons.price.idx': (92.2, 94.7, 93.7),
    'cons.conf.idx': (-50.8, -26.9, -41.8),
    'euribor3m': (0.6, 5.05, 4.8),
    'nr.employed': (4963, 5228, 5191)
}

# Features not found in range files but present in column_names.txt 
# Assuming these are binary/engineered features that need manual input
EXTRA_FEATURES = ['contacted_before', 'previous_success', 'is_success_month', 'euribor_low']

@st.cache_resource
def load_model():
    try:
        with open(MODEL_FILE, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found. Please ensure `{MODEL_FILE}` is in the same directory.")
        return None

model = load_model()

# ==========================================
# 2. UI LAYOUT & INPUTS
# ==========================================
st.title("🏦 Bank Marketing Subscription Predictor")
st.markdown("Enter client details below to predict the likelihood of subscribing to a term deposit.")

# Form for user input
with st.form("prediction_form"):
    
    # --- Group 1: Client Profile ---
    st.subheader("👤 Client Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=NUMERICAL_RANGES['age'][0], max_value=NUMERICAL_RANGES['age'][1], value=NUMERICAL_RANGES['age'][2])
        job = st.selectbox("Job", CATEGORICAL_OPTIONS['job'])
        marital = st.selectbox("Marital Status", CATEGORICAL_OPTIONS['marital'])
    with c2:
        education = st.selectbox("Education", CATEGORICAL_OPTIONS['education'])
        default = st.selectbox("Has Credit in Default?", CATEGORICAL_OPTIONS['default'])
    with c3:
        housing = st.selectbox("Has Housing Loan?", CATEGORICAL_OPTIONS['housing'])
        loan = st.selectbox("Has Personal Loan?", CATEGORICAL_OPTIONS['loan'])

    st.markdown("---")

    # --- Group 2: Campaign & Contact Info ---
    st.subheader("📞 Current Campaign & Contact")
    c4, c5, c6 = st.columns(3)
    with c4:
        contact = st.selectbox("Contact Communication Type", CATEGORICAL_OPTIONS['contact'])
        month = st.selectbox("Last Contact Month", CATEGORICAL_OPTIONS['month'])
    with c5:
        day_of_week = st.selectbox("Last Contact Day", CATEGORICAL_OPTIONS['day_of_week'])
        campaign = st.number_input("Contacts during this campaign", min_value=1, max_value=60, value=2)
    with c6:
        # Extra feature assumption: is_success_month
        is_success_month = st.selectbox("Is Historic Success Month?", [0, 1], help="Derived feature: Is this month historically successful?")

    st.markdown("---")

    # --- Group 3: History & Socio-Economics ---
    st.subheader("📊 History & Socio-Economic Indicators")
    c7, c8, c9 = st.columns(3)
    with c7:
        pdays = st.number_input("Days since last contact (999=Never)", min_value=0, max_value=999, value=999)
        previous = st.number_input("Number of contacts before this campaign", min_value=0, max_value=10, value=0)
        poutcome = st.selectbox("Outcome of previous campaign", CATEGORICAL_OPTIONS['poutcome'])
        # Extra features
        contacted_before = st.selectbox("Contacted Before?", [0, 1])
        previous_success = st.selectbox("Previous Success?", [0, 1])
        
    with c8:
        emp_var_rate = st.number_input("Employment Variation Rate", value=NUMERICAL_RANGES['emp.var.rate'][2])
        cons_price_idx = st.number_input("Consumer Price Index", value=NUMERICAL_RANGES['cons.price.idx'][2])
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=NUMERICAL_RANGES['cons.conf.idx'][2])
        
    with c9:
        euribor3m = st.number_input("Euribor 3 Month Rate", value=NUMERICAL_RANGES['euribor3m'][2])
        nr_employed = st.number_input("Number of Employees", value=NUMERICAL_RANGES['nr.employed'][2])
        # Extra feature
        euribor_low = st.selectbox("Is Euribor Low?", [0, 1], help="Derived feature: Is the rate historically low?")

    submit_button = st.form_submit_button("Predict Subscription")

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if submit_button and model is not None:
    # 1. Prepare Data Dictionary mapping inputs to column_names.txt 
    input_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
        # Derived features must be passed as they are in the column list
        'contacted_before': contacted_before,
        'previous_success': previous_success,
        'is_success_month': is_success_month,
        'euribor_low': euribor_low
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # 2. Make Prediction
    try:
        prediction = model.predict(df_input)[0]
        # Handle cases where model might return probabilities or raw thresholds
        
        st.markdown("---")
        st.subheader("Results")
        
        if prediction == 1:
            st.success("## ✅ Prediction: Client will SUBSCRIBE")
        else:
            st.error("## ❌ Prediction: Client will NOT SUBSCRIBE")

      # ==========================================
        # 4. SHAP VISUALIZATION (XGBOOST OPTIMIZED)
        # ==========================================
        st.subheader("🔍 Model Explanation (SHAP)")
        
        try:
            with st.spinner('Analyzing XGBoost feature importance...'):
                # 1. Unwrap TunedThresholdClassifierCV
                # The 'estimator_' attribute holds the underlying Pipeline
                inner_pipeline = model.estimator_ if hasattr(model, 'estimator_') else model
                
                # 2. Extract Preprocessor and XGBoost Classifier
                # We assume 'columntransformer' and 'classifier' are the step names
                preprocessor = inner_pipeline.named_steps['columntransformer']
                xgb_model = inner_pipeline.named_steps['classifier']
                
                # 3. Transform the single row of input data
                # This converts categories to one-hot/label encoding as per your training
                transformed_data = preprocessor.transform(df_input)
                
                # 4. Initialize TreeExplainer for XGBoost
                # We pass the underlying XGBoost booster for better compatibility
                explainer = shap.TreeExplainer(xgb_model)
                
                # 5. Calculate SHAP values for the transformed input
                shap_values = explainer.shap_values(transformed_data)
                
                # 6. Plotting
                st.set_option('deprecation.showPyplotGlobalUse', False)
                
                # If XGBoost returns a 2D array for binary classification, 
                # we select the values for the positive class (subscription)
                if isinstance(shap_values, list):
                    display_shap = shap_values[1]
                elif len(shap_values.shape) == 3: # Some versions return (samples, features, classes)
                    display_shap = shap_values[:, :, 1]
                else:
                    display_shap = shap_values

                # Get feature names from the preprocessor to make the chart readable
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    feature_names = [f"Feature {i}" for i in range(transformed_data.shape[1])]

                fig = shap.force_plot(
                    explainer.expected_value, 
                    display_shap[0], 
                    features=transformed_data[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig, bbox_inches='tight')

        except Exception as e:
            st.error(f"SHAP Extraction Error: {str(e)}")
            st.info("Check if your pipeline steps are exactly named 'columntransformer' and 'classifier'.")
