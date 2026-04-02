import streamlit as st
import numpy as np
import joblib

# Load saved model & scaler
model = joblib.load("heart_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Heart Disease Prediction App")

st.write("Enter patient details:")

# Example inputs (adjust based on your dataset columns)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0-2)", [0, 1, 2])

# Convert input into array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak,
                        slope, ca, thal]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")