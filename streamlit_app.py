import streamlit as st

st.title("Bienvenue Abdirahman 👋")

st.header("Je découvre Streamlit")

nom = st.text_input("Entrez votre nom")

if st.button("Valider"):
    st.success(f"Bonjour {nom} 🚀")

age = st.slider("Choisissez votre âge", 0, 100, 20)

st.write("Votre âge est :", age)
