import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="🏥 Prédiction du Diabète",
    page_icon="🩺",
    layout="centered"
)

# ============================================================
# CHARGEMENT DU MODÈLE DIABÈTE
# ============================================================
@st.cache_resource
def load_model(fichier_pkl):
    data = pickle.load(fichier_pkl)
    return (
        data['models'],
        data['imputer'],
        data['scaler'],
        data['features'],
        data['best']
    )

# ============================================================
# TITRE
# ============================================================
st.title("🩺 Prédiction du Diabète")
st.write("Renseignez vos données médicales et le modèle prédit si vous êtes diabétique ou non.")
st.markdown("---")

# ============================================================
# SIDEBAR — Chargement modèle + infos
# ============================================================
with st.sidebar:
    st.header("📁 Charger le modèle")
    pkl_file = st.file_uploader("Uploader modele_diabete.pkl", type="pkl")

    st.markdown("---")
    st.header("ℹ️ Informations")
    st.write("**Modèles disponibles :**")
    st.write("• 🌲 Random Forest")
    st.write("• 🚀 Gradient Boosting")
    st.write("• 📈 Régression Logistique")
    st.markdown("---")
    st.write("**Variables utilisées :**")
    variables_info = {
        "Pregnancies"              : "🤰 Grossesses",
        "Glucose"                  : "🩸 Glucose (mg/dL)",
        "BloodPressure"            : "💓 Tension (mmHg)",
        "SkinThickness"            : "📏 Épaisseur peau (mm)",
        "Insulin"                  : "💉 Insuline (µU/ml)",
        "BMI"                      : "⚖️ IMC (kg/m²)",
        "DiabetesPedigreeFunction" : "🧬 Pedigree Diabète",
        "Age"                      : "🎂 Âge (ans)",
    }
    for v in variables_info.values():
        st.write(f"• {v}")
    st.markdown("---")
    st.write("**Maladies détectables :**")
    st.write("• ✅ Non Diabétique")
    st.write("• ⚠️ Diabétique")

# ============================================================
# VÉRIFICATION MODÈLE
# ============================================================
if pkl_file is None:
    st.info("👈 **Chargez votre fichier `modele_diabete.pkl`** dans la barre latérale pour commencer.")
    st.stop()

# Chargement
try:
    models, imputer, scaler, features, best_name = load_model(pkl_file)
    st.sidebar.success(f"✅ Modèle chargé !\n🥇 Meilleur : **{best_name}**")
    st.sidebar.write(f"**Nombre de variables :** {len(features)}")
    st.sidebar.write(f"**Nombre de classes :** 2 (Diabétique / Non)")
except Exception as e:
    st.error(f"❌ Erreur de chargement : {e}")
    st.stop()

# ============================================================
# SAISIE DES DONNÉES PATIENT
# ============================================================
st.subheader("🩺 Renseignez vos données médicales :")

col1, col2 = st.columns(2)

with col1:
    age         = st.slider("🎂 Âge (ans)", 18, 90, 33)
    pregnancies = st.slider("🤰 Nombre de grossesses", 0, 20, 3)
    bmi         = st.slider("⚖️ IMC (kg/m²)", 10.0, 70.0, 32.0, 0.1)
    skin        = st.slider("📏 Épaisseur pli cutané (mm)", 0, 100, 23)

with col2:
    glucose = st.slider("🩸 Glucose (mg/dL)", 50, 250, 117)
    insulin = st.slider("💉 Insuline (µU/ml)", 0, 900, 30)
    bp      = st.slider("💓 Tension diastolique (mmHg)", 30, 130, 72)
    dpf     = st.slider("🧬 Score Pedigree Diabète", 0.05, 2.50, 0.37, 0.01)

st.markdown("---")

# Choix du modèle
model_choice = st.selectbox(
    "🤖 Choisir le modèle :",
    list(models.keys()),
    index=list(models.keys()).index(best_name),
    help=f"Modèle recommandé : {best_name}"
)

st.markdown("---")

# ============================================================
# BOUTON DE PRÉDICTION
# ============================================================
if st.button("🔍 Prédire le Diabète", use_container_width=True):

    if glucose == 0 and bmi == 0:
        st.warning("⚠️ Veuillez renseigner des valeurs correctes !")

    else:
        # Construction du vecteur patient
        patient = pd.DataFrame(
            [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=features
        )

        # Preprocessing
        patient_imp = imputer.transform(patient)
        patient_sc  = scaler.transform(patient_imp)

        # Prédiction avec le modèle choisi
        model      = models[model_choice]
        prediction = model.predict(patient_sc)[0]
        probas     = model.predict_proba(patient_sc)[0]

        # Résultat principal
        if prediction == 1:
            st.error(f"### ⚠️ Résultat : **DIABÉTIQUE**")
        else:
            st.success(f"### ✅ Résultat : **NON DIABÉTIQUE**")

        st.markdown("---")

        # Top 2 probabilités (comme le Top 3 du code original)
        st.subheader("📊 Probabilités :")

        classes   = ["Non Diabétique", "Diabétique"]
        top2_idx  = np.argsort(probas)[::-1][:2]

        for i, idx in enumerate(top2_idx):
            emoji = "🥇" if i == 0 else "🥈"
            label = classes[idx]
            prob  = probas[idx]
            st.write(f"{emoji} **{label}** — {prob*100:.1f}%")
            st.progress(float(prob))

        st.markdown("---")

        # Top 3 modèles (comme la version originale compare plusieurs maladies)
        st.subheader("🤖 Avis des 3 modèles :")

        for nom, mod in models.items():
            pred_m  = mod.predict(patient_sc)[0]
            proba_m = mod.predict_proba(patient_sc)[0]
            result  = "⚠️ Diabétique" if pred_m == 1 else "✅ Non Diabétique"
            badge   = "🥇 " if nom == best_name else ""
            st.write(f"{badge}**{nom}** → {result} "
                     f"({proba_m[1]*100:.1f}% diabète)")
            st.progress(float(proba_m[1]))

        st.markdown("---")
        st.info("⚠️ Ce résultat est indicatif uniquement. Consultez un médecin pour un diagnostic officiel.")
