
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="CardioCare AI", page_icon="🫀", layout="wide")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('heart_failure_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_assets()

# --- 3. UI LAYOUT ---
st.title("🫀 CardioCare: Heart Failure Risk Predictor")
st.markdown("### Clinical Decision Support System")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Vitals")
    
    # Numerical Inputs (Sliders for typical medical ranges)
    age = st.slider("Age", 40, 95, 60)
    
    c1, c2 = st.columns(2)
    with c1:
        creatinine_phosphokinase = st.number_input("CPK (mcg/L)", 20, 8000, 582)
        ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", 25000, 850000, 260000)
    with c2:
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.1, step=0.1)
        serum_sodium = st.slider("Serum Sodium (mEq/L)", 110, 150, 137)
        time = 0 # Dummy variable not used in prediction but kept for structure

    st.subheader("Clinical History")
    # Categorical Inputs (0 or 1)
    anaemia = st.checkbox("Anaemia")
    diabetes = st.checkbox("Diabetes")
    high_blood_pressure = st.checkbox("High Blood Pressure")
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    smoking = st.checkbox("Smoker")

with col2:
    st.subheader("Risk Analysis")
    
    if st.button("Analyze Patient Risk", type="primary"):
        if model is not None:
            # 1. Prepare Input Array
            # Convert booleans/strings to 0/1
            sex_val = 1 if sex == "Male" else 0
            
            # The order MUST match the training columns exactly:
            # age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
            # high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking
            
            features = np.array([[
                age, 
                int(anaemia), 
                creatinine_phosphokinase, 
                int(diabetes), 
                ejection_fraction, 
                int(high_blood_pressure), 
                platelets, 
                serum_creatinine, 
                serum_sodium, 
                sex_val, 
                int(smoking)
            ]])
            
            # 2. Scale the Input
            features_scaled = scaler.transform(features)
            
            # 3. Predict
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1] # Prob of Class 1 (Death)
            
            # 4. Visualization
            risk_color = "#ff4b4b" if probability > 0.5 else "#00c0f2"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Mortality Risk (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(0, 192, 242, 0.2)"},
                        {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                    ],
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Interpretation
            if prediction == 1:
                st.error("⚠️ **High Risk Detected:** The model predicts a potential adverse event.")
                st.markdown("**Critical Factors to Watch:**")
                if ejection_fraction < 40: st.write("- Low Ejection Fraction")
                if serum_creatinine > 1.5: st.write("- Elevated Serum Creatinine")
            else:
                st.success("✅ **Low Risk:** The model predicts patient survival during follow-up.")

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only. Not for medical diagnosis.")
