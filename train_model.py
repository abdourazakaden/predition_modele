import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionimport streamlit as st
import pandas as pd
import numpy as np

# ========================
# TITRE PRINCIPAL
# ========================


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
import pickle

# Charger données
df = pd.read_csv("train.csv")

# Nettoyage simple
df = df.dropna()

# Séparer X et y
X = df.drop("special_monitoring_flag", axis=1)
y = df["special_monitoring_flag"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Sauvegarder modèle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle sauvegardé ✅")
