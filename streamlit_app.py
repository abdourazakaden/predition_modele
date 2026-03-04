import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="MediPredict — Diabète AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS GLOBAL — Design médical premium sombre
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #060a12 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8eaf0;
}

[data-testid="stSidebar"] {
    background: #0b1120 !important;
    border-right: 1px solid #1a2540;
}

/* Masquer les éléments Streamlit */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060a12; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d2044 50%, #071830 100%);
    border: 1px solid #1a3560;
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,180,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,255,170,0.05) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,255,0.12);
    border: 1px solid rgba(0,180,255,0.3);
    color: #00b4ff;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a8c8ff 60%, #00e5ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 16px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #7a8fa8;
    font-weight: 300;
    max-width: 600px;
    line-height: 1.7;
}
.hero-stats {
    display: flex;
    gap: 40px;
    margin-top: 32px;
}
.hero-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00e5ff;
}
.hero-stat-lbl {
    font-size: 0.78rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Metric Cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
.metric-card {
    background: #0b1120;
    border: 1px solid #1a2540;
    border-radius: 16px;
    padding: 24px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #2a4570; }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg, #00b4ff, #00e5ff));
}
.metric-icon { font-size: 1.5rem; margin-bottom: 12px; }
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--color, #00e5ff);
    line-height: 1;
}
.metric-label { font-size: 0.8rem; color: #4a6080; margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }
.metric-delta { font-size: 0.78rem; color: #2ecc71; margin-top: 4px; }

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 20px;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1a2540, transparent);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3a5070;
}

/* ── Input Panel ── */
.input-panel {
    background: #0b1120;
    border: 1px solid #1a2540;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 16px;
}
.input-panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3a7bd5;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #1a2540;
}

/* ── Result Cards ── */
.result-positive {
    background: linear-gradient(135deg, #0a2818, #0d3520);
    border: 1px solid #1a6640;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #1a0a0a, #2d1010);
    border: 1px solid #661a1a;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}
.result-icon { font-size: 3rem; margin-bottom: 12px; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
}
.result-sub { font-size: 0.9rem; color: #7a8fa8; margin-top: 8px; }

/* ── Proba Cards ── */
.proba-card {
    background: #0b1120;
    border: 1px solid #1a2540;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.proba-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
}
.proba-label { font-size: 0.8rem; color: #4a6080; margin-top: 8px; text-transform: uppercase; letter-spacing: 1px; }

/* ── Model Compare Cards ── */
.model-card {
    background: #0b1120;
    border: 1px solid #1a2540;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s;
}
.model-card.best {
    border-color: #00b4ff;
    background: linear-gradient(135deg, #0b1120, #0d1e35);
}
.model-card-name { font-size: 0.8rem; color: #4a6080; margin-bottom: 12px; }
.model-card-auc {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
}
.model-card-acc { font-size: 0.85rem; color: #7a8fa8; margin-top: 6px; }
.best-badge {
    display: inline-block;
    background: rgba(0,180,255,0.15);
    border: 1px solid rgba(0,180,255,0.4);
    color: #00b4ff;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 10px;
    margin-bottom: 8px;
}

/* ── Importance Bar ── */
.imp-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.imp-label { font-size: 0.85rem; color: #a0b0c8; width: 180px; flex-shrink: 0; }
.imp-bar-bg {
    flex: 1;
    height: 8px;
    background: #1a2540;
    border-radius: 4px;
    overflow: hidden;
}
.imp-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #1e5fa8, #00b4ff);
}
.imp-pct { font-size: 0.8rem; color: #00b4ff; font-weight: 600; width: 45px; text-align: right; }

/* ── Info Box ── */
.info-box {
    background: rgba(0, 180, 255, 0.05);
    border: 1px solid rgba(0, 180, 255, 0.15);
    border-left: 3px solid #00b4ff;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.88rem;
    color: #8aa8c8;
    line-height: 1.7;
    margin: 16px 0;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0b1120;
    border-radius: 12px;
    padding: 6px;
    border: 1px solid #1a2540;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #4a6080 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1a3560 !important;
    color: #a8d4ff !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #1e5fa8, #00b4ff) !important;
}
.stSlider label { font-size: 0.85rem !important; color: #7a8fa8 !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] label { color: #7a8fa8 !important; font-size: 0.85rem !important; }
[data-testid="stSelectbox"] > div > div {
    background: #0f1e35 !important;
    border: 1px solid #1a3560 !important;
    border-radius: 10px !important;
    color: #e0eaff !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #0b1120 !important;
    border: 1px dashed #1a3560 !important;
    border-radius: 14px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] label { color: #4a6080 !important; font-size: 0.85rem !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a4fa8, #0099dd) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(0, 153, 221, 0.25) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2060c0, #00b4ff) !important;
    box-shadow: 0 6px 30px rgba(0, 153, 221, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a8d4ff, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.sidebar-tagline { font-size: 0.72rem; color: #2a4060; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 24px; }
.sidebar-section {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2a4060;
    margin: 20px 0 10px;
}
.nav-item {
    background: rgba(26, 53, 96, 0.3);
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #7a8fa8;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a2540 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Success / Error / Info ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid #1a3560 !important;
}

/* ── Caption ── */
.stCaption { color: #2a4060 !important; font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
@st.cache_resource
def charger_modele(fichier):
    return pickle.load(fichier)

labels_fr = {
    'Pregnancies'             : 'Grossesses',
    'Glucose'                 : 'Glucose',
    'BloodPressure'           : 'Tension artérielle',
    'SkinThickness'           : 'Épaisseur peau',
    'Insulin'                 : 'Insuline',
    'BMI'                     : 'IMC',
    'DiabetesPedigreeFunction': 'Pedigree Diabète',
    'Age'                     : 'Âge',
}

scores_ref = {
    'Régression Logistique': {'acc': 0.7338, 'auc': 0.8126, 'cv': 0.8415},
    'Random Forest':         {'acc': 0.7468, 'auc': 0.8267, 'cv': 0.8262},
    'Gradient Boosting':     {'acc': 0.7403, 'auc': 0.8057, 'cv': 0.7924},
}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">MediPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Diabète · Intelligence Artificielle</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">📁 Modèle ML</div>', unsafe_allow_html=True)
    pkl_file = st.file_uploader("Charger modele_diabete.pkl", type="pkl", label_visibility="collapsed")

    if pkl_file:
        st.markdown("""
        <div style="background:rgba(0,180,100,0.08);border:1px solid rgba(0,180,100,0.25);
        border-radius:10px;padding:10px 14px;font-size:0.8rem;color:#40c080;margin-top:8px">
        ✅ Modèle chargé avec succès
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📊 Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-item">🗂️ Pima Indians Diabetes</div>
    <div class="nav-item">👥 768 patients · 8 variables</div>
    <div class="nav-item">📍 Source : Kaggle</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🤖 Modèles disponibles</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-item">🌲 Random Forest — AUC 0.827</div>
    <div class="nav-item">🚀 Gradient Boosting — AUC 0.806</div>
    <div class="nav-item">📈 Régression Logistique — AUC 0.813</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">ℹ️ Version</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-item">v2.0 · Mars 2025</div>
    """, unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">🩺 Plateforme IA Médicale</div>
    <div class="hero-title">Prédiction du Diabète<br>par Intelligence Artificielle</div>
    <div class="hero-sub">
        Analysez les données médicales d'un patient et obtenez instantanément
        une prédiction de risque diabétique basée sur des modèles de Machine Learning
        entraînés sur 768 cas réels.
    </div>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-val">82.7%</div>
            <div class="hero-stat-lbl">AUC Score</div>
        </div>
        <div>
            <div class="hero-stat-val">768</div>
            <div class="hero-stat-lbl">Patients</div>
        </div>
        <div>
            <div class="hero-stat-val">3</div>
            <div class="hero-stat-lbl">Modèles ML</div>
        </div>
        <div>
            <div class="hero-stat-val">8</div>
            <div class="hero-stat-lbl">Variables</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# VÉRIFICATION MODÈLE
# ============================================================
if pkl_file is None:
    st.markdown("""
    <div class="info-box">
    👈 <strong>Commencez par charger votre fichier <code>modele_diabete.pkl</code></strong> dans la barre latérale gauche.<br>
    Ce fichier est généré par le script d'entraînement <code>train_modele_colab.py</code> dans Google Colab.
    </div>
    """, unsafe_allow_html=True)

    # Étapes d'utilisation
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Comment utiliser</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, num, icon, titre, desc in zip(
        [col1, col2, col3, col4],
        ["01", "02", "03", "04"],
        ["☁️", "📥", "⬆️", "🔮"],
        ["Entraîner", "Télécharger", "Uploader", "Prédire"],
        ["Exécuter train_modele_colab.py dans Google Colab",
         "Récupérer le fichier modele_diabete.pkl généré",
         "Charger le .pkl dans la sidebar de cette app",
         "Saisir les données patient et obtenir le résultat"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#1e5fa8,#00b4ff)">
                <div style="font-family:'Syne',sans-serif;font-size:0.65rem;color:#2a4060;
                letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">ÉTAPE {num}</div>
                <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                color:#a8d4ff;margin-bottom:8px">{titre}</div>
                <div style="font-size:0.8rem;color:#4a6080;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# Chargement
try:
    data      = charger_modele(pkl_file)
    models    = data['models']
    imputer   = data['imputer']
    scaler    = data['scaler']
    features  = data['features']
    best_name = data['best']
except Exception as e:
    st.error(f"❌ Erreur de chargement : {e}")
    st.stop()

# ============================================================
# MÉTRIQUES GLOBALES
# ============================================================
st.markdown("""
<div class="section-header">
    <div class="section-title">Performance globale</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

best_scores = scores_ref[best_name]
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card" style="--accent:linear-gradient(90deg,#1e5fa8,#00b4ff);--color:#00e5ff">
        <div class="metric-icon">🥇</div>
        <div class="metric-value">{best_scores['auc']:.3f}</div>
        <div class="metric-label">AUC — Meilleur Modèle</div>
        <div class="metric-delta">↑ {best_name}</div>
    </div>
    <div class="metric-card" style="--accent:linear-gradient(90deg,#1e8040,#00c860);--color:#00c860">
        <div class="metric-icon">🎯</div>
        <div class="metric-value">{best_scores['acc']*100:.1f}%</div>
        <div class="metric-label">Accuracy (Test 20%)</div>
        <div class="metric-delta">↑ Données réelles</div>
    </div>
    <div class="metric-card" style="--accent:linear-gradient(90deg,#805010,#ffaa00);--color:#ffaa00">
        <div class="metric-icon">🔄</div>
        <div class="metric-value">{best_scores['cv']:.3f}</div>
        <div class="metric-label">CV AUC (5-Fold)</div>
        <div class="metric-delta">↑ Score robuste</div>
    </div>
    <div class="metric-card" style="--accent:linear-gradient(90deg,#601e80,#cc44ff);--color:#cc44ff">
        <div class="metric-icon">🤖</div>
        <div class="metric-value">3</div>
        <div class="metric-label">Modèles disponibles</div>
        <div class="metric-delta">↑ Comparaison en temps réel</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ONGLETS PRINCIPAUX
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "  🔮  Prédiction Patient  ",
    "  📊  Performance & Analyse  ",
    "  🏆  Variables Importantes  ",
])

# ╔══════════════════════════════════════════════════════════╗
# ║              TAB 1 — PRÉDICTION PATIENT                  ║
# ╚══════════════════════════════════════════════════════════╝
with tab1:

    st.markdown("""
    <div class="section-header">
        <div class="section-title">Données du patient</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="input-panel"><div class="input-panel-title">👤 Profil Personnel</div>', unsafe_allow_html=True)
        age         = st.slider("Âge (ans)", 18, 90, 33)
        pregnancies = st.slider("Nombre de grossesses", 0, 20, 3)
        bmi         = st.slider("IMC — Indice de Masse Corporelle (kg/m²)", 10.0, 70.0, 32.0, 0.1)
        skin        = st.slider("Épaisseur du pli cutané (mm)", 0, 100, 23)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="input-panel"><div class="input-panel-title">🩸 Analyses Biologiques</div>', unsafe_allow_html=True)
        glucose = st.slider("Glucose plasmatique (mg/dL)", 50, 250, 117)
        insulin = st.slider("Insuline sérique 2h (µU/ml)", 0, 900, 30)
        bp      = st.slider("Tension diastolique (mmHg)", 30, 130, 72)
        dpf     = st.slider("Score Pedigree Diabète", 0.05, 2.50, 0.37, 0.01)
        st.markdown('</div>', unsafe_allow_html=True)

    # Modèle + bouton
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Paramètres de prédiction</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_sel, col_btn = st.columns([1, 1], gap="large")
    with col_sel:
        model_choice = st.selectbox(
            "Modèle ML",
            list(models.keys()),
            index=list(models.keys()).index(best_name)
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  ANALYSER LE PATIENT", use_container_width=True)

    # ── Résultat ──
    if predict_btn:
        patient     = pd.DataFrame(
            [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=features
        )
        patient_imp = imputer.transform(patient)
        patient_sc  = scaler.transform(patient_imp)

        model = models[model_choice]
        pred  = model.predict(patient_sc)[0]
        proba = model.predict_proba(patient_sc)[0]

        st.markdown("""
        <div class="section-header">
            <div class="section-title">Résultat de l'analyse</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col_res, col_p1, col_p2 = st.columns([2, 1, 1], gap="large")

        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="result-negative">
                    <div class="result-icon">⚠️</div>
                    <div class="result-label" style="color:#ff5555">DIABÈTE DÉTECTÉ</div>
                    <div class="result-sub">Risque élevé — Consultation médicale recommandée</div>
                    <div style="margin-top:16px;font-size:0.75rem;color:#4a3030;
                    letter-spacing:1px;text-transform:uppercase">
                        Modèle utilisé : {model_choice}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-positive">
                    <div class="result-icon">✅</div>
                    <div class="result-label" style="color:#40d080">PAS DE DIABÈTE</div>
                    <div class="result-sub">Aucun signe détecté selon ce modèle</div>
                    <div style="margin-top:16px;font-size:0.75rem;color:#1a4030;
                    letter-spacing:1px;text-transform:uppercase">
                        Modèle utilisé : {model_choice}
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_p1:
            st.markdown(f"""
            <div class="proba-card">
                <div class="proba-value" style="color:#40d080">{proba[0]*100:.1f}%</div>
                <div class="proba-label">✅ Non Diabétique</div>
                <div style="margin-top:12px;height:4px;background:#1a2540;border-radius:2px;overflow:hidden">
                    <div style="width:{proba[0]*100}%;height:100%;background:linear-gradient(90deg,#1e8040,#40d080)"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_p2:
            st.markdown(f"""
            <div class="proba-card">
                <div class="proba-value" style="color:#ff5555">{proba[1]*100:.1f}%</div>
                <div class="proba-label">⚠️ Diabétique</div>
                <div style="margin-top:12px;height:4px;background:#1a2540;border-radius:2px;overflow:hidden">
                    <div style="width:{proba[1]*100}%;height:100%;background:linear-gradient(90deg,#801a1a,#ff5555)"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Graphique probabilité
        st.markdown("""
        <div class="section-header">
            <div class="section-title">Analyse visuelle</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col_chart, col_recap = st.columns([1, 1], gap="large")

        with col_chart:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#0b1120')
            ax.set_facecolor('#0b1120')
            cats  = ['Non Diabétique', 'Diabétique']
            vals  = [proba[0]*100, proba[1]*100]
            cols  = ['#40d080', '#ff5555']
            bars  = ax.barh(cats, vals, color=cols, height=0.4, alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(min(val+1, 88), bar.get_y()+bar.get_height()/2,
                        f'{val:.1f}%', va='center', color='white',
                        fontweight='bold', fontsize=14,
                        fontfamily='DejaVu Sans')
            ax.set_xlim(0, 110)
            ax.axvline(50, color='#2a4060', linestyle='--', lw=1.5, alpha=0.7)
            ax.text(50.5, 1.35, 'Seuil 50%', color='#2a4060', fontsize=8)
            ax.set_xlabel('Probabilité (%)', color='#4a6080', fontsize=10)
            ax.tick_params(colors='#4a6080', labelsize=10)
            ax.set_title('Probabilités de Prédiction', color='#7a9fc8',
                         fontsize=12, fontweight='bold', pad=14)
            for s in ax.spines.values(): s.set_edgecolor('#1a2540')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_recap:
            # Récapitulatif patient
            valeurs = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
            unites  = ['grossesses', 'mg/dL', 'mmHg', 'mm', 'µU/ml', 'kg/m²', 'score', 'ans']
            st.markdown("""
            <div style="background:#0b1120;border:1px solid #1a2540;border-radius:16px;padding:20px">
                <div style="font-family:'Syne',sans-serif;font-size:0.7rem;font-weight:600;
                letter-spacing:2px;text-transform:uppercase;color:#2a5080;margin-bottom:16px">
                Profil Patient
                </div>
            """, unsafe_allow_html=True)
            for f, v, u in zip(features, valeurs, unites):
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 0;border-bottom:1px solid #0f1e35">
                    <div style="font-size:0.83rem;color:#5a7090">{labels_fr.get(f, f)}</div>
                    <div style="font-size:0.9rem;color:#a8d4ff;font-weight:500">{v} <span style="color:#2a4060;font-size:0.75rem">{u}</span></div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Avis des 3 modèles
        st.markdown("""
        <div class="section-header">
            <div class="section-title">Consensus des 3 modèles</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        cols3 = st.columns(3, gap="medium")
        for i, (nom, mod) in enumerate(models.items()):
            p  = mod.predict(patient_sc)[0]
            pr = mod.predict_proba(patient_sc)[0]
            is_best  = nom == best_name
            res_color = "#ff5555" if p == 1 else "#40d080"
            res_label = "⚠️ DIABÉTIQUE" if p == 1 else "✅ NON DIABÉTIQUE"
            with cols3[i]:
                st.markdown(f"""
                <div class="model-card {'best' if is_best else ''}">
                    {"<div class='best-badge'>MEILLEUR</div>" if is_best else ""}
                    <div class="model-card-name">{nom}</div>
                    <div style="font-size:1.05rem;font-weight:700;color:{res_color};
                    margin:8px 0">{res_label}</div>
                    <div style="display:flex;justify-content:space-around;margin-top:14px">
                        <div style="text-align:center">
                            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;
                            font-weight:700;color:#40d080">{pr[0]*100:.0f}%</div>
                            <div style="font-size:0.7rem;color:#2a4060">Non Diab.</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;
                            font-weight:700;color:#ff5555">{pr[1]*100:.0f}%</div>
                            <div style="font-size:0.7rem;color:#2a4060">Diabète</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        ⚠️ <strong>Avertissement médical</strong> — Cette application est à usage éducatif et de recherche uniquement.
        Les résultats ne constituent pas un diagnostic médical. Consultez toujours un professionnel de santé qualifié.
        </div>
        """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║           TAB 2 — PERFORMANCE & ANALYSE                  ║
# ╚══════════════════════════════════════════════════════════╝
with tab2:

    st.markdown("""
    <div class="section-header">
        <div class="section-title">Comparaison des modèles</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # Cards modèles
    cols_m = st.columns(3, gap="medium")
    for i, (nom, s) in enumerate(scores_ref.items()):
        is_best = nom == best_name
        color   = "#00e5ff" if is_best else "#3a7bd5"
        with cols_m[i]:
            st.markdown(f"""
            <div class="model-card {'best' if is_best else ''}">
                {"<div class='best-badge'>RECOMMANDÉ</div>" if is_best else ""}
                <div class="model-card-name">{nom}</div>
                <div class="model-card-auc" style="color:{color}">{s['auc']:.3f}</div>
                <div style="font-size:0.7rem;color:#2a4060;margin-bottom:12px">AUC Score</div>
                <div style="display:flex;justify-content:space-around;
                border-top:1px solid #1a2540;padding-top:12px;margin-top:4px">
                    <div style="text-align:center">
                        <div style="font-size:1rem;font-weight:600;color:#a8d4ff">{s['acc']*100:.1f}%</div>
                        <div style="font-size:0.68rem;color:#2a4060;text-transform:uppercase;letter-spacing:1px">Accuracy</div>
                    </div>
                    <div style="text-align:center">
                        <div style="font-size:1rem;font-weight:600;color:#a8d4ff">{s['cv']:.3f}</div>
                        <div style="font-size:0.68rem;color:#2a4060;text-transform:uppercase;letter-spacing:1px">CV AUC</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <div class="section-title">Graphiques de performance</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2, gap="large")

    with col_g1:
        # Barres comparaison
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#0b1120')
        ax.set_facecolor('#0b1120')
        noms  = [n.replace(' ', '\n') for n in scores_ref.keys()]
        accs  = [s['acc'] for s in scores_ref.values()]
        aucs  = [s['auc'] for s in scores_ref.values()]
        cvs   = [s['cv']  for s in scores_ref.values()]
        x     = np.arange(3)
        b1 = ax.bar(x-0.25, accs, 0.22, label='Accuracy',  color='#1e5fa8', alpha=0.9)
        b2 = ax.bar(x,       aucs, 0.22, label='AUC Test',  color='#00b4ff', alpha=0.9)
        b3 = ax.bar(x+0.25,  cvs,  0.22, label='CV AUC',    color='#40d080', alpha=0.9)
        for bars in [b1, b2, b3]:
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f'{bar.get_height():.3f}', ha='center',
                        color='white', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(noms, color='#7a8fa8', fontsize=9)
        ax.set_ylim(0.65, 0.95)
        ax.set_ylabel('Score', color='#4a6080')
        ax.set_title('Comparaison — Accuracy / AUC / CV AUC',
                     color='#7a9fc8', fontweight='bold', fontsize=12, pad=14)
        ax.legend(facecolor='#0d1e35', labelcolor='#7a8fa8', fontsize=9,
                  framealpha=0.8, edgecolor='#1a2540')
        ax.tick_params(colors='#4a6080')
        for s in ax.spines.values(): s.set_edgecolor('#1a2540')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        # Explication métriques
        st.markdown("""
        <div style="background:#0b1120;border:1px solid #1a2540;border-radius:16px;padding:24px;height:100%">
            <div style="font-family:'Syne',sans-serif;font-size:0.7rem;font-weight:600;
            letter-spacing:2px;text-transform:uppercase;color:#2a5080;margin-bottom:20px">
            Interprétation des métriques
            </div>

            <div style="margin-bottom:16px;padding:14px;background:#0f1e35;border-radius:10px;border-left:3px solid #00b4ff">
                <div style="font-weight:600;color:#a8d4ff;font-size:0.9rem">AUC Score</div>
                <div style="font-size:0.8rem;color:#5a7090;margin-top:4px;line-height:1.6">
                Capacité du modèle à distinguer diabétiques / non diabétiques.<br>
                1.0 = parfait · 0.5 = aléatoire
                </div>
            </div>

            <div style="margin-bottom:16px;padding:14px;background:#0f1e35;border-radius:10px;border-left:3px solid #40d080">
                <div style="font-weight:600;color:#a8d4ff;font-size:0.9rem">Accuracy</div>
                <div style="font-size:0.8rem;color:#5a7090;margin-top:4px;line-height:1.6">
                Pourcentage de prédictions correctes sur les données de test (20% du dataset).
                </div>
            </div>

            <div style="padding:14px;background:#0f1e35;border-radius:10px;border-left:3px solid #ffaa00">
                <div style="font-weight:600;color:#a8d4ff;font-size:0.9rem">CV AUC (5-Fold)</div>
                <div style="font-size:0.8rem;color:#5a7090;margin-top:4px;line-height:1.6">
                AUC moyen calculé sur 5 sous-ensembles. Mesure la robustesse et généralisation du modèle.
                </div>
            </div>

            <div style="margin-top:20px;padding:12px 14px;background:rgba(0,180,255,0.05);
            border:1px solid rgba(0,180,255,0.15);border-radius:10px">
                <div style="font-size:0.78rem;color:#4a6080">
                📊 AUC entre 0.80 et 0.90 → Bon niveau pour un diagnostic médical ✅
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║           TAB 3 — VARIABLES IMPORTANTES                  ║
# ╚══════════════════════════════════════════════════════════╝
with tab3:

    st.markdown("""
    <div class="section-header">
        <div class="section-title">Importance des variables — Random Forest</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    rf  = models['Random Forest']
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    icons = {
        'Glucose':'🩸', 'BMI':'⚖️', 'Age':'🎂',
        'DiabetesPedigreeFunction':'🧬', 'Insulin':'💉',
        'BloodPressure':'💓', 'SkinThickness':'📏', 'Pregnancies':'🤰'
    }
    medals = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣']

    col_bars, col_cards = st.columns([3, 2], gap="large")

    with col_bars:
        # Barres HTML stylées
        st.markdown("""
        <div style="background:#0b1120;border:1px solid #1a2540;border-radius:16px;padding:28px">
        """, unsafe_allow_html=True)

        for i, (feat, val) in enumerate(imp.items()):
            pct   = val * 100
            width = val / imp.max() * 100
            icon  = icons.get(feat, '📋')
            lbl   = labels_fr.get(feat, feat)
            gradient = f"linear-gradient(90deg, #1e5fa8, #00b4ff)" if i == 0 else \
                       f"linear-gradient(90deg, #155a30, #40d080)" if i == 1 else \
                       f"linear-gradient(90deg, #5a3010, #ffaa00)" if i == 2 else \
                       f"linear-gradient(90deg, #1a2540, #2a4570)"
            st.markdown(f"""
            <div class="imp-row">
                <div style="width:200px;flex-shrink:0;display:flex;align-items:center;gap:8px">
                    <span style="font-size:1rem">{icon}</span>
                    <span class="imp-label">{lbl}</span>
                </div>
                <div class="imp-bar-bg">
                    <div class="imp-bar-fill" style="width:{width}%;background:{gradient}"></div>
                </div>
                <div class="imp-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_cards:
        for i, (feat, val) in enumerate(imp.items()):
            icon  = icons.get(feat, '📋')
            lbl   = labels_fr.get(feat, feat)
            colors_card = ['#00e5ff','#40d080','#ffaa00','#a855f7',
                           '#f97316','#ec4899','#64748b','#94a3b8']
            color = colors_card[i]
            st.markdown(f"""
            <div style="background:#0b1120;border:1px solid #1a2540;border-radius:14px;
            padding:14px 18px;margin-bottom:8px;display:flex;align-items:center;
            justify-content:space-between;border-left:3px solid {color}">
                <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:1.2rem">{medals[i]}</span>
                    <div>
                        <div style="font-size:0.9rem;color:#a8d4ff;font-weight:500">{icon} {lbl}</div>
                    </div>
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                font-weight:700;color:{color}">{val*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Explications
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Pourquoi ces variables ?</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    expl = [
        ("🩸", "Glucose", "#00e5ff",
         "Indicateur principal du diabète. Un taux élevé de glucose dans le sang (hyperglycémie) est le symptôme direct du diabète. C'est la variable la plus discriminante."),
        ("⚖️", "IMC", "#40d080",
         "L'obésité est le principal facteur de risque du diabète de type 2. Plus l'IMC est élevé, plus la résistance à l'insuline augmente."),
        ("🎂", "Âge", "#ffaa00",
         "Le risque de diabète augmente avec l'âge. Le métabolisme du glucose se dégrade progressivement et la sensibilité à l'insuline diminue."),
        ("🧬", "Pedigree Diabète", "#a855f7",
         "Ce score mesure la probabilité génétique basée sur l'historique familial. Un pedigree élevé indique des antécédents familiaux de diabète."),
    ]

    cols_expl = st.columns(2, gap="medium")
    for i, (icon, titre, color, desc) in enumerate(expl):
        with cols_expl[i % 2]:
            st.markdown(f"""
            <div style="background:#0b1120;border:1px solid #1a2540;border-radius:14px;
            padding:20px;margin-bottom:12px;border-top:2px solid {color}">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                    <span style="font-size:1.4rem">{icon}</span>
                    <div style="font-family:'Syne',sans-serif;font-size:1rem;
                    font-weight:700;color:{color}">{titre}</div>
                </div>
                <div style="font-size:0.85rem;color:#5a7090;line-height:1.7">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="margin-top:48px;padding:24px 0;border-top:1px solid #1a2540;
text-align:center;display:flex;justify-content:space-between;align-items:center">
    <div style="font-family:'Syne',sans-serif;font-weight:700;
    background:linear-gradient(135deg,#a8d4ff,#00e5ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    MediPredict
    </div>
    <div style="font-size:0.75rem;color:#2a4060">
    Dataset Pima Indians Diabetes · Kaggle · Random Forest · Gradient Boosting · Régression Logistique
    </div>
    <div style="font-size:0.75rem;color:#2a4060">v2.0 · 2025</div>
</div>
""", unsafe_allow_html=True)
