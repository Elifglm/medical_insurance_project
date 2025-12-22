import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. LOAD MODELS
@st.cache_resource
def load_models():
    reg = joblib.load('regresyon_model.pkl')
    clf = joblib.load('siniflandirma_model.pkl')
    return reg, clf

try:
    reg_model, clf_model = load_models()
    reg_columns = list(reg_model.feature_names_in_)
    clf_columns = list(clf_model.feature_names_in_)
except Exception as e:
    st.error(f"Model Loading Error: {e}")

# Page Setup
st.set_page_config(page_title="AI Health Analysis", layout="wide")
st.title("🛡️ AI-Powered Medical Cost & Risk Analysis")

# 2. USER INPUTS (In English)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographic Details")
    age = st.number_input("Age", 18, 100, 35)
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 24.5)
    income = st.number_input("Annual Income ($)", 0, 1000000, 55000)
    sex = st.selectbox("Gender", ["Male", "Female", "Other"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])

with col2:
    st.subheader("🏥 Clinical Background")
    avg_claim = st.number_input("Average Past Claim ($)", 0.0, 100000.0, 2500.0)
    smoker = st.selectbox("Smoking Status", ["Never", "Former", "Smoker"])
    alcohol = st.selectbox("Alcohol Frequency", ["Occasional", "Weekly", "Unknown"])
    medication = st.slider("Daily Medication Count", 0, 20, 1)

with col3:
    st.subheader("🎓 Socio-Economic Info")
    education = st.selectbox("Education", ["HS", "Some College", "Masters", "Doctorate", "No HS"])
    employment = st.selectbox("Employment", ["Self-employed", "Retired", "Unemployed", "Employed"])
    urban_rural = st.selectbox("Area Type", ["Urban", "Suburban", "Rural"])

# 3. ANALYTICS ENGINE
if st.button("RUN PREDICTION"):
    # Create common pool
    all_needed_cols = list(set(reg_columns + clf_columns))
    input_df = pd.DataFrame(0, index=[0], columns=all_needed_cols)
    
    # Assign Numeric Values
    input_df['age'] = age
    input_df['bmi'] = bmi
    input_df['income'] = income
    input_df['avg_claim_amount'] = avg_claim
    input_df['medication_count'] = medication
    
    # Clinical defaults for hidden features
    for tech_col, val in zip(['systolic_bp', 'ldl', 'hba1c'], [120, 100, 5.4]):
        if tech_col in input_df.columns:
            input_df[tech_col] = val

    # Encoding Mapping
    def set_dummy(df, prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in df.columns:
            df[col_name] = 1

    set_dummy(input_df, "sex", sex)
    set_dummy(input_df, "region", region)
    set_dummy(input_df, "smoker", smoker)
    set_dummy(input_df, "education", education)
    set_dummy(input_df, "employment_status", employment)
    set_dummy(input_df, "urban_rural", urban_rural)
    set_dummy(input_df, "alcohol_freq", alcohol)

    # Clean person_id issue
    if 'person_id' in input_df.columns:
        input_df['person_id'] = 0

    # Model specific alignment
    input_reg = input_df[reg_columns]
    input_clf = input_df[clf_columns]

    # PREDICTIONS
    try:
        cost_val = reg_model.predict(input_reg)[0]
        risk_val = clf_model.predict(input_clf)[0]

        # 4. RESULTS DISPLAY
        st.divider()
        res1, res2 = st.columns(2)
        
        with res1:
            st.metric("Predicted Annual Cost", f"${cost_val:,.2f}")
            st.caption("Metric: Regression R-Squared (66%)")

        with res2:
            risk_label = "HIGH RISK" if risk_val == 1 else "LOW RISK"
            color = "red" if risk_val == 1 else "green"
            st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_label}</span>", unsafe_allow_html=True)
            st.caption("Metric: Classification Accuracy (79%)")
            
        st.success("Analysis Complete!")
    except Exception as e:
        st.error(f"Prediction Error: {e}")