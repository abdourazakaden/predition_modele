import streamlit as st
import pickle
import numpy as np

# Charger modèle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Prédiction Special Monitoring 🚀")

st.header("Entrez les informations")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
income = st.number_input("Income", min_value=0.0, value=5.0)
credit_amount = st.number_input("Credit Amount", min_value=0.0, value=5.0)

if st.button("Prédire"):

    data = np.array([[age, income, credit_amount]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Client à surveiller")
    else:
        st.success("✅ Client normal")
