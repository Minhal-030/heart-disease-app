import os
import pickle
import pandas as pd
import streamlit as st

st.title("Heart Disease Prediction System")
st.write("Enter the patient details below and click Predict.")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
model = pickle.load(open(model_path, "rb"))

# Input fields (the 4 selected attributes)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest pain type (1 to 4)", [1, 2, 3, 4])
thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=150)

if st.button("Predict"):
    patient = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "cp": [cp],
        "thalach": [thalach],
    })
    result = model.predict(patient)[0]
    if result == 1:
        st.error("Prediction: Heart disease likely")
    else:
        st.success("Prediction: Heart disease unlikely")
