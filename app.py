import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ROBUST ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        # Get the absolute path of the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct full paths to the model files
        # This fixes the "File not found" error by looking in the 'models' folder
        # or the root folder automatically.
        
        # Check if models are in a 'models' subdirectory (Best Practice)
        model_path = os.path.join(script_dir, 'models', 'heart_failure_model.pkl')
        scaler_path = os.path.join(script_dir, 'models', 'scaler.pkl')
        
        # Fallback: Check root directory if not found in 'models'
        if not os.path.exists(model_path):
            model_path = os.path.join(script_dir, 'heart_failure_model.pkl')
            scaler_path = os.path.join(script_dir, 'scaler.pkl')

        if not os.path.exists(model_path):
            st.error(f"❌ Critical Error: Model file not found at {model_path}")
            return None, None

        # Load the files
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error loading assets: {e}")
        return None, None

model, scaler = load_assets()

# --- 3. SIDEBAR INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("CardioCare AI")
    st.markdown("### Clinical Decision Support")
    st.info(
        "This tool uses a Support Vector Machine (SVM) algorithm to estimate mortality risk "
        "in heart failure patients based on 11 clinical features."
    )
    st.markdown("---")
    st.caption("v1.0.0 | Built for Day 9")

# --- 4. MAIN INTERFACE ---
st.title("🫀 CardioCare: Heart Failure Risk Assessment")
st.markdown("Enter patient clinical records below to generate a risk profile.")

if model is None:
    st.warning("⚠️ Models not loaded. Please run 'python train.py' to generate model files.")
    st.stop()

# Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Patient Vitals & History")
    
    # Group 1: Demographics & History
    with st.expander("Demographics & History", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 40, 95, 60, help="Patient age in years")
            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        with c2:
            smoking = st.checkbox("Smoker")
            diabetes = st.checkbox("Diabetes")
            high_blood_pressure = st.checkbox("High Blood Pressure")
            anaemia = st.checkbox("Anaemia")

    # Group 2: Lab Results
    with st.expander("Clinical Metrics (Lab Results)", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            creatinine_phosphokinase = st.number_input("CPK Level (mcg/L)", 20, 8000, 582, help="Level of the CPK enzyme in the blood")
            platelets = st.number_input("Platelets (k/mL)", 25000.0, 850000.0, 260000.0, step=1000.0)
        with c4:
            serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.1, step=0.1, help="Level of serum creatinine in the blood")
            serum_sodium = st.slider("Serum Sodium (mEq/L)", 110, 150, 137)
    
    # Group 3: Cardiac Function
    with st.expander("Cardiac Function", expanded=True):
        ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38, help="Percentage of blood leaving the heart at each contraction")

with col2:
    st.subheader("📊 Risk Analysis Report")
    
    # Analyze Button
    analyze_btn = st.button("Generate Risk Profile", type="primary", use_container_width=True)
    
    if analyze_btn:
        # 1. Prepare Input Array (Order MUST match training data exactly)
        # Columns: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
        # high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking
        
        sex_val = 1 if sex == "Male" else 0
        
        input_data = np.array([[
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
        
        # 2. Scale Features
        try:
            input_scaled = scaler.transform(input_data)
            
            # 3. Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1] # Probability of Death (Class 1)
            
            # 4. Display Gauge Chart
            risk_color = "#FF4B4B" if probability > 0.5 else "#00C0F2"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Mortality Risk Probability", 'font': {'size': 24}},
                number = {'suffix': "%", 'font': {'color': risk_color}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(0, 192, 242, 0.1)"},
                        {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.1)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Clinical Interpretation
            st.markdown("### Clinical Interpretation")
            if prediction == 1:
                st.error("⚠️ **High Risk Group:** The model predicts a potential adverse event during the follow-up period.")
                st.markdown("**Key Risk Factors Identified:**")
                
                # Simple logic to highlight common risk drivers
                if ejection_fraction < 35:
                    st.write("- **Low Ejection Fraction:** Heart pumping efficiency is critically low.")
                if serum_creatinine > 1.5:
                    st.write("- **Elevated Creatinine:** Signs of potential renal dysfunction.")
                if age > 70:
                    st.write("- **Advanced Age:** Increases baseline risk.")
            else:
                st.success("✅ **Low Risk Group:** The model predicts patient survival during the follow-up period.")
                st.write("Patient vitals are within stable ranges relative to the training cohort.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- 5. FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 12px;'>
        ⚠️ <b>Disclaimer:</b> This application is a Clinical Decision Support System (CDSS) prototype. 
        It is NOT a certified medical device and should not be used for primary diagnosis. 
        Always consult a medical professional.
    </div>
    """, 
    unsafe_allow_html=True
                                     )
