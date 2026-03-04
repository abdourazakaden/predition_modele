import streamlit as st
import pandas as pd
import numpy as np

# ========================
# TITRE PRINCIPAL
# ========================
st.title("Application Streamlit - Niveau Débutant 🚀")

st.header("1️⃣ Texte et affichage")

st.write("Bienvenue Abdirahman 👋")
st.markdown("**Streamlit** est simple et puissant.")

# ========================
# INPUT UTILISATEUR
# ========================
st.header("2️⃣ Interaction utilisateur")

nom = st.text_input("Entrez votre nom :")

age = st.slider("Choisissez votre âge :", 0, 100, 20)

if st.button("Valider"):
    st.success(f"Bonjour {nom}, vous avez {age} ans !")

# ========================
# SIDEBAR
# ========================
st.sidebar.title("Menu")
choix = st.sidebar.selectbox(
    "Choisissez une option",
    ["Accueil", "Données", "Graphique"]
)

# ========================
# CONTENU SELON MENU
# ========================
if choix == "Accueil":
    st.write("Bienvenue sur la page d'accueil.")

elif choix == "Données":
    st.subheader("Tableau de données")
    df = pd.DataFrame({
        "Colonne A": np.random.randn(10),
        "Colonne B": np.random.randn(10)
    })
    st.dataframe(df)

elif choix == "Graphique":
    st.subheader("Graphique simple")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["A", "B", "C"]
    )
    st.line_chart(chart_data)
