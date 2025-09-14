import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("climate_risk_rf_model.pkl")

# Title
st.title("🔥 Climate Risk Prediction App")
st.subheader("Predict Fire Risk Based on Environmental Features")

# Define input fields dynamically (example features — replace with actual ones)
feature_names = ['Temperature', 'Humidity', 'WindSpeed', 'Rainfall', 'SoilMoisture']  # Update as needed
user_input = {}

st.sidebar.header("Input Features")
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict Fire Risk"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.success(f"Prediction: {'🔥 Fire Risk' if prediction == 1 else '✅ No Fire Risk'}")
    st.write(f"Confidence: {np.max(proba)*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("Built by Bhanuteja Naresh Samal | Climate Risk Modeling Project")
