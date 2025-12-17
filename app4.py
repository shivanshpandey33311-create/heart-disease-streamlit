import streamlit as st
import pandas as pd
import pickle 

with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

st.title("Heart Disease Prediction")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
bp = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxhr = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": cp,
        "RestingBP": bp,
        "Cholesterol": chol,
        "FastingBS": fbs,
        "RestingECG": ecg,
        "MaxHR": maxhr,
        "ExerciseAngina": exang,
        "Oldpeak": oldpeak,
        "ST_Slope": slope
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)

    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")