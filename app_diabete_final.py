import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, roc_curve)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="🩺 Prédiction Diabète",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; }
    .card {
        background: #1a1d27;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3142;
        margin-bottom: 10px;
    }
    .card-value { font-size: 2.2rem; font-weight: bold; }
    .card-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .result-box {
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .diabete    { background: #3d1515; border: 2px solid #e74c3c; color: #e74c3c; }
    .no-diabete { background: #0d3321; border: 2px solid #2ecc71; color: #2ecc71; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT & ENTRAÎNEMENT (mis en cache)
# ============================================================
@st.cache_resource
def entrainer_modeles(data_bytes):
    import io
    df = pd.read_csv(io.BytesIO(data_bytes), sep=';')

    # Remplacer les 0 impossibles par NaN
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_zero] = df[cols_zero].replace(0, np.nan)

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[features]
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc  = scaler.transform(X_test_imp)

    models = {
        'Gradient Boosting':     GradientBoostingClassifier(n_estimators=200, random_state=42),
        'Random Forest':         RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'Régression Logistique': LogisticRegression(max_iter=1000, class_weight='balanced'),
    }
    for model in models.values():
        model.fit(X_train_sc, y_train)

    return models, imputer, scaler, features, X_test_sc, y_test, df

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://img.icons8.com/color/96/diabetes.png", width=80)
st.sidebar.title("🩺 Diabète Predictor")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("📁 Charger diabete1.csv", type="csv")

st.sidebar.markdown("---")
st.sidebar.info("""
**Variables utilisées :**
- 🤰 Grossesses
- 🩸 Glucose
- 💓 Tension artérielle
- 📏 Épaisseur peau
- 💉 Insuline
- ⚖️ IMC
- 🧬 Pedigree Diabète
- 🎂 Âge
""")

# ============================================================
# TITRE
# ============================================================
st.title("🩺 Application de Prédiction du Diabète")
st.markdown("Modèle entraîné sur le **dataset Pima Indians Diabetes** — 768 patients réels")
st.markdown("---")

if uploaded is None:
    st.info("👈 **Chargez le fichier `diabete1.csv`** dans la barre latérale pour commencer.")
    st.markdown("""
    ### 📋 À propos de cette application
    Cette application utilise le **Machine Learning** pour prédire si une personne
    est diabétique ou non, basée sur des mesures médicales.

    ### 🔬 Données utilisées
    Le dataset contient **768 patientes** avec 8 variables médicales :
    | Variable | Description |
    |---|---|
    | Pregnancies | Nombre de grossesses |
    | Glucose | Concentration de glucose (mg/dL) |
    | BloodPressure | Pression diastolique (mmHg) |
    | SkinThickness | Épaisseur du pli cutané (mm) |
    | Insulin | Insuline sérique 2h (µU/ml) |
    | BMI | Indice de masse corporelle |
    | DiabetesPedigreeFunction | Score historique familial |
    | Age | Âge (ans) |
    """)
    st.stop()

# Entraînement
with st.spinner("⚙️ Entraînement des modèles..."):
    data_bytes = uploaded.read()
    models, imputer, scaler, features, X_test_sc, y_test, df = entrainer_modeles(data_bytes)

st.success("✅ Modèles entraînés avec succès sur vos données !")
st.markdown("---")

