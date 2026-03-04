import streamlit as st
import pandas as pd
import numpy as np

# ============================================
# TITRE PRINCIPAL
# ============================================
st.title("🎓 Streamlit - Niveau 1 : Les Bases")
st.markdown("---")

# ============================================
# SECTION 1 : TEXTE
# ============================================
st.header("📝 1. Les Types de Texte")

st.write("Ceci est un texte simple avec `st.write()`")
st.markdown("**Gras**, *italique*, et `code` avec `st.markdown()`")
st.success("✅ Message de succès")
st.error("❌ Message d'erreur")
st.info("ℹ️ Message d'information")
st.warning("⚠️ Message d'avertissement")

st.markdown("---")

# ============================================
# SECTION 2 : INTERACTIONS
# ============================================
st.header("🎛️ 2. Les Interactions")

# Texte input
nom = st.text_input("👤 Entrez votre prénom :")
if nom:
    st.success(f"Bonjour **{nom}** ! 👋")

# Slider
age = st.slider("🎂 Votre âge :", 1, 100, 20)
st.write(f"Vous avez **{age} ans**")

# Case à cocher
accord = st.checkbox("✔️ J'accepte les conditions")
if accord:
    st.info("Merci d'avoir accepté !")

# Bouton
if st.button("🎲 Cliquez ici !"):
    st.balloons()
    st.write("Vous avez cliqué ! 🎉")

st.markdown("---")

# ============================================
# SECTION 3 : TABLEAU
# ============================================
st.header("📋 3. Tableau de Données")

data = pd.DataFrame({
    "Nom"  : ["Alice", "Bob", "Charlie", "Diana"],
    "Age"  : [25, 30, 35, 28],
    "Ville": ["Paris", "Lyon", "Marseille", "Bordeaux"]
})
st.dataframe(data)

st.markdown("---")

# ============================================
# SECTION 4 : GRAPHIQUES
# ============================================
st.header("📊 4. Graphiques")

valeurs = np.random.randn(30)

st.subheader("Graphique en ligne")
st.line_chart(valeurs)

st.subheader("Graphique en barres")
st.bar_chart(valeurs)

st.markdown("---")
st.markdown("✅ **Félicitations ! Vous avez terminé le Niveau 1 !**")
