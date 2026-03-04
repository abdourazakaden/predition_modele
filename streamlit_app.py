import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
    page_title="Prédiction Diabète",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1a1d27; border-radius: 12px;
        padding: 16px; text-align: center; border: 1px solid #333;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #2ecc71; }
    .metric-label { font-size: 0.85rem; color: #aaa; }
    .stTabs [data-baseweb="tab"] { color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT & PRÉPARATION DES DONNÉES
# ============================================================
@st.cache_data
def charger_et_entrainer():
    train = pd.read_csv('train.csv')
    val   = pd.read_csv('validation.csv')
    test  = pd.read_csv('test.csv')

    def preparer(df):
        df = df[df['DIQ010'].isin([1.0, 2.0])].copy()
        df['target'] = (df['DIQ010'] == 1.0).astype(int)
        return df

    train, val, test = preparer(train), preparer(val), preparer(test)

    DROP = ['SEQN','DIQ010','target','BPXCHR','LBDLDL','LBXGLT','LBXTR','SMQ040']
    features = [c for c in train.columns if c not in DROP]

    X_train, y_train = train[features], train['target']
    X_val,   y_val   = val[features],   val['target']
    X_test,  y_test  = test[features],  test['target']

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc   = scaler.transform(X_val_imp)
    X_test_sc  = scaler.transform(X_test_imp)

    models = {
        'Régression Logistique': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest':         RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'Gradient Boosting':     GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    for model in models.values():
        model.fit(X_train_sc, y_train)

    return models, imputer, scaler, features, X_val_sc, y_val, X_test_sc, y_test, train, val, test

# ============================================================
# CHARGEMENT DES FICHIERS
# ============================================================
st.sidebar.title("🩺 Diabète ML App")
st.sidebar.markdown("---")

st.sidebar.subheader("📁 Chargement des données")
f_train = st.sidebar.file_uploader("train.csv",      type="csv")
f_val   = st.sidebar.file_uploader("validation.csv", type="csv")
f_test  = st.sidebar.file_uploader("test.csv",       type="csv")

# ============================================================
# TITRE
# ============================================================
st.title("🩺 Prédiction du Diabète — Machine Learning")
st.markdown("Analyse complète basée sur les données **NHANES** (enquête nationale de santé américaine)")
st.markdown("---")

if not (f_train and f_val and f_test):
    st.info("👈 **Veuillez charger les 3 fichiers CSV** dans la barre latérale pour commencer.")
    st.markdown("""
    ### 📋 Fichiers requis :
    - `train.csv` — données d'entraînement
    - `validation.csv` — données de validation
    - `test.csv` — données de test

    ### 🎯 Objectif :
    Prédire si une personne est **diabétique** ou **non diabétique** à partir de :
    - données démographiques (âge, genre, ethnie)
    - mesures corporelles (IMC, poids, taille, tour de taille)
    - valeurs biologiques (cholestérol, hémoglobine...)
    - habitudes de vie (alimentation, activité physique)
    """)
    st.stop()

# Sauvegarde temporaire pour le cache
import tempfile, os, shutil

with tempfile.TemporaryDirectory() as tmpdir:
    for fname, fobj in [('train.csv', f_train), ('validation.csv', f_val), ('test.csv', f_test)]:
        path = os.path.join(tmpdir, fname)
        with open(path, 'wb') as out:
            out.write(fobj.read())
        shutil.copy(path, fname)

with st.spinner("⚙️ Entraînement des modèles en cours..."):
    models, imputer, scaler, features, X_val_sc, y_val, X_test_sc, y_test, df_train, df_val, df_test = charger_et_entrainer()

st.success("✅ Modèles entraînés avec succès !")
st.markdown("---")

# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue d'ensemble",
    "🏆 Performance des Modèles",
    "📈 Visualisations",
    "🔮 Prédiction Individuelle"
])

# ----------------------------------------
# TAB 1 : VUE D'ENSEMBLE
# ----------------------------------------
with tab1:
    st.header("📊 Vue d'Ensemble des Données")

    col1, col2, col3, col4 = st.columns(4)
    total = len(df_train) + len(df_val) + len(df_test)
    diab  = int(df_train['target'].sum() + df_val['target'].sum() + df_test['target'].sum())
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total patients</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#e74c3c">{diab}</div>
            <div class="metric-label">Diabétiques</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#3498db">{total - diab}</div>
            <div class="metric-label">Non Diabétiques</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#f39c12">{len(features)}</div>
            <div class="metric-label">Variables utilisées</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Répartition des données")
        repartition = pd.DataFrame({
            'Dataset': ['Train', 'Validation', 'Test'],
            'Total': [len(df_train), len(df_val), len(df_test)],
            'Diabétiques': [int(df_train['target'].sum()), int(df_val['target'].sum()), int(df_test['target'].sum())],
        })
        repartition['Non Diabétiques'] = repartition['Total'] - repartition['Diabétiques']
        repartition['% Diabète'] = (repartition['Diabétiques'] / repartition['Total'] * 100).round(1)
        st.dataframe(repartition, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📊 Distribution des classes (Train)")
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        counts = df_train['target'].value_counts()
        bars = ax.bar(['Non Diabétique', 'Diabétique'], counts.values,
                      color=['#3498db', '#e74c3c'], alpha=0.85, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(int(bar.get_height())), ha='center', color='white', fontweight='bold')
        ax.set_ylabel('Nombre', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🔍 Aperçu des données d'entraînement")
    st.dataframe(df_train[features].head(10), use_container_width=True)

# ----------------------------------------
# TAB 2 : PERFORMANCE
# ----------------------------------------
with tab2:
    st.header("🏆 Performance des Modèles")

    resultats = {}
    for nom, model in models.items():
        y_pred_v = model.predict(X_val_sc)
        y_prob_v = model.predict_proba(X_val_sc)[:,1]
        y_pred_t = model.predict(X_test_sc)
        y_prob_t = model.predict_proba(X_test_sc)[:,1]
        resultats[nom] = {
            'acc_val':  accuracy_score(y_val, y_pred_v),
            'auc_val':  roc_auc_score(y_val, y_prob_v),
            'acc_test': accuracy_score(y_test, y_pred_t),
            'auc_test': roc_auc_score(y_test, y_prob_t),
            'y_pred_t': y_pred_t,
            'y_prob_t': y_prob_t,
        }

    best_name = max(resultats, key=lambda x: resultats[x]['auc_test'])

    # Tableau comparatif
    df_res = pd.DataFrame({
        'Modèle': list(resultats.keys()),
        'Accuracy Val': [f"{resultats[m]['acc_val']:.3f}" for m in resultats],
        'AUC Val':      [f"{resultats[m]['auc_val']:.3f}" for m in resultats],
        'Accuracy Test':[f"{resultats[m]['acc_test']:.3f}" for m in resultats],
        'AUC Test':     [f"{resultats[m]['auc_test']:.3f}" for m in resultats],
    })
    st.subheader("📋 Tableau Comparatif")
    st.dataframe(df_res, use_container_width=True, hide_index=True)
    st.success(f"🥇 **Meilleur modèle : {best_name}** (AUC Test = {resultats[best_name]['auc_test']:.3f})")

    st.markdown("---")
    st.subheader(f"📋 Rapport Détaillé — {best_name}")

    report = classification_report(
        y_test, resultats[best_name]['y_pred_t'],
        target_names=['Non Diabétique', 'Diabétique'],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    # Matrice de confusion
    st.markdown("---")
    st.subheader("🔲 Matrice de Confusion")
    cm = confusion_matrix(y_test, resultats[best_name]['y_pred_t'])
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')
    im = ax.imshow(cm, cmap='Blues')
    labels = ['Non Diabétique', 'Diabétique']
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Prédit: Non', 'Prédit: Oui'], color='white')
    ax.set_yticklabels(['Réel: Non', 'Réel: Oui'], color='white')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white', fontsize=20, fontweight='bold')
    ax.set_title(f'Matrice de Confusion\n({best_name})', color='white', fontweight='bold')
    st.pyplot(fig)
    plt.close()

# ----------------------------------------
# TAB 3 : VISUALISATIONS
# ----------------------------------------
with tab3:
    st.header("📈 Visualisations")

    col1, col2 = st.columns(2)

    # Courbes ROC
    with col1:
        st.subheader("Courbes ROC")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        colors_roc = ['#3498db', '#2ecc71', '#e74c3c']
        for (nom, res), color in zip(resultats.items(), colors_roc):
            fpr, tpr, _ = roc_curve(y_test, res['y_prob_t'])
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{nom.split()[0]} ({res['auc_test']:.3f})")
        ax.plot([0,1],[0,1],'w--', alpha=0.4)
        ax.set_xlabel('Faux Positifs', color='white')
        ax.set_ylabel('Vrais Positifs', color='white')
        ax.set_title('Courbes ROC', color='white', fontweight='bold')
        ax.legend(fontsize=9, facecolor='#252835', labelcolor='white')
        ax.tick_params(colors='white')
        for s in ax.spines.values(): s.set_edgecolor('#333')
        st.pyplot(fig)
        plt.close()

    # Comparaison barres
    with col2:
        st.subheader("Comparaison des scores")
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')
        noms = [n.replace(' ', '\n') for n in resultats.keys()]
        aucs_test = [resultats[m]['auc_test'] for m in resultats]
        accs_test = [resultats[m]['acc_test'] for m in resultats]
        x = np.arange(len(noms))
        b1 = ax.bar(x - 0.2, accs_test, 0.35, label='Accuracy', color='#3498db', alpha=0.85)
        b2 = ax.bar(x + 0.2, aucs_test,  0.35, label='AUC',      color='#2ecc71', alpha=0.85)
        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f'{bar.get_height():.3f}', ha='center', color='white', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(noms, color='white', fontsize=8)
        ax.set_ylim(0.6, 1.05)
        ax.set_title('Accuracy vs AUC (Test)', color='white', fontweight='bold')
        ax.legend(facecolor='#252835', labelcolor='white')
        ax.tick_params(colors='white')
        for s in ax.spines.values(): s.set_edgecolor('#333')
        st.pyplot(fig)
        plt.close()

    # Feature importance
    st.markdown("---")
    st.subheader("🏆 Variables les plus importantes (Random Forest)")
    rf_model = models['Random Forest']
    labels_map = {
        'RIDAGEYR':'Âge', 'BMXBMI':'IMC', 'BMXWAIST':'Tour de taille',
        'BMXWT':'Poids', 'BMXHT':'Taille', 'INDFMPIR':'Revenu/Pauvreté',
        'BPXSY1':'Tension syst. 1', 'BPXSY2':'Tension syst. 2',
        'BPXDI1':'Tension dia. 1', 'LBXTC':'Cholestérol total',
        'LBDHDD':'HDL Cholestérol', 'LBXHGB':'Hémoglobine',
        'LBXSAL':'Albumine', 'DR1TKCAL':'Calories', 'DR1TPROT':'Protéines',
        'DR1TCARB':'Glucides', 'RIAGENDR':'Genre', 'RIDRETH1':'Ethnie',
        'DMDEDUC2':'Éducation', 'DMDMARTL':'Statut marital',
        'PAQ650':'Activité physique', 'ALQ101':'Alcool',
        'HEQ010':'Hépatite', 'WHQ030':'Poids perçu',
    }
    imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False).head(12)
    nice = [labels_map.get(f, f) for f in imp.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')
    bar_c = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp)))[::-1]
    bars = ax.bar(range(len(imp)), imp.values, color=bar_c, alpha=0.9)
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels(nice, rotation=35, ha='right', color='white', fontsize=9)
    ax.set_ylabel('Importance', color='white')
    ax.tick_params(colors='white')
    ax.set_title('Top 12 Variables Importantes', color='white', fontweight='bold')
    for s in ax.spines.values(): s.set_edgecolor('#333')
    st.pyplot(fig)
    plt.close()

# ----------------------------------------
# TAB 4 : PRÉDICTION INDIVIDUELLE
# ----------------------------------------
with tab4:
    st.header("🔮 Prédiction pour un Nouveau Patient")
    st.write("Entrez les données d'un patient pour obtenir une prédiction.")

    model_choice = st.selectbox("🤖 Choisir le modèle :", list(models.keys()))

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Démographie")
        genre  = st.selectbox("Genre", [("Masculin", 1), ("Féminin", 2)], format_func=lambda x: x[0])[1]
        age    = st.slider("Âge (ans)", 18, 80, 45)
        ethnie = st.selectbox("Ethnie", [
            ("Mexicain-Américain", 1), ("Autre Hispanique", 2),
            ("Blanc non-Hispanique", 3), ("Noir non-Hispanique", 4), ("Autre", 5)
        ], format_func=lambda x: x[0])[1]
        education = st.selectbox("Niveau d'éducation", [
            ("Moins que lycée", 1), ("Lycée", 2), ("Bac+2", 3), ("Licence", 4), ("Master+", 5)
        ], format_func=lambda x: x[0])[1]
        mariage = st.selectbox("Statut marital", [
            ("Marié(e)", 1), ("Veuf(ve)", 2), ("Divorcé(e)", 3),
            ("Séparé(e)", 4), ("Célibataire", 5), ("Concubinage", 6)
        ], format_func=lambda x: x[0])[1]
        revenu = st.slider("Ratio revenu/pauvreté", 0.0, 5.0, 2.0, 0.1)

    with col2:
        st.subheader("📏 Mesures Corporelles")
        bmi      = st.slider("IMC (kg/m²)", 15.0, 60.0, 28.0, 0.1)
        poids    = st.slider("Poids (kg)", 30.0, 200.0, 75.0, 0.5)
        taille   = st.slider("Taille (cm)", 140.0, 200.0, 165.0, 0.5)
        waist    = st.slider("Tour de taille (cm)", 50.0, 180.0, 90.0, 0.5)
        st.subheader("💉 Tension Artérielle")
        bpsy1 = st.slider("Systolique 1 (mmHg)", 80, 220, 120)
        bpsy2 = st.slider("Systolique 2 (mmHg)", 80, 220, 118)
        bpsy3 = st.slider("Systolique 3 (mmHg)", 80, 220, 116)
        bpdi1 = st.slider("Diastolique 1 (mmHg)", 40, 130, 78)
        bpdi2 = st.slider("Diastolique 2 (mmHg)", 40, 130, 76)
        bpdi3 = st.slider("Diastolique 3 (mmHg)", 40, 130, 74)

    with col3:
        st.subheader("🧬 Biologie")
        cholesterol = st.slider("Cholestérol total (mg/dL)", 100, 400, 200)
        hdl         = st.slider("HDL Cholestérol (mg/dL)", 20, 120, 50)
        hemoglobine = st.slider("Hémoglobine (g/dL)", 8.0, 20.0, 14.0, 0.1)
        albumine    = st.slider("Albumine sérique (g/dL)", 2.0, 6.0, 4.2, 0.1)
        st.subheader("🍽️ Alimentation")
        calories = st.slider("Calories (kcal/j)", 500, 5000, 2000)
        proteines = st.slider("Protéines (g/j)", 10.0, 300.0, 80.0, 0.5)
        glucides  = st.slider("Glucides (g/j)", 50.0, 600.0, 250.0, 0.5)
        lipides   = st.slider("Lipides (g/j)", 10.0, 300.0, 70.0, 0.5)
        st.subheader("🏃 Mode de Vie")
        grossesse = st.selectbox("Grossesse", [("Non", 2), ("Oui", 1)], format_func=lambda x: x[0])[1]
        activite  = st.selectbox("Activité physique", [("Oui", 1), ("Non", 2)], format_func=lambda x: x[0])[1]
        alcool    = st.selectbox("Consommation alcool", [("Oui", 1), ("Non", 2)], format_func=lambda x: x[0])[1]
        hepatite  = st.selectbox("Hépatite", [("Non", 2), ("Oui", 1)], format_func=lambda x: x[0])[1]
        poids_percu = st.selectbox("Poids perçu", [
            ("Trop maigre", 1), ("Normal", 2), ("Surpoids", 3)
        ], format_func=lambda x: x[0])[1]

    st.markdown("---")

    if st.button("🔮 Lancer la Prédiction", use_container_width=True):
        patient = pd.DataFrame([[
            genre, age, ethnie, education, grossesse, revenu, mariage,
            calories, proteines, glucides, lipides,
            bmi, poids, taille, bpsy1, bpsy2, bpsy3, bpdi1, bpdi2, bpdi3,
            waist, cholesterol, hdl, hemoglobine, albumine,
            alcool, activite, hepatite, poids_percu
        ]], columns=features)

        patient_imp = imputer.transform(patient)
        patient_sc  = scaler.transform(patient_imp)

        model_sel = models[model_choice]
        proba = model_sel.predict_proba(patient_sc)[0]
        pred  = model_sel.predict(patient_sc)[0]

        st.markdown("---")
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if pred == 1:
                st.error(f"### ⚠️ Résultat : DIABÉTIQUE")
                st.error(f"Probabilité d'être diabétique : **{proba[1]*100:.1f}%**")
            else:
                st.success(f"### ✅ Résultat : NON DIABÉTIQUE")
                st.success(f"Probabilité d'être non diabétique : **{proba[0]*100:.1f}%**")

        with col_r2:
            # Jauge de probabilité
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('#1a1d27')
            ax.set_facecolor('#1a1d27')
            color = '#e74c3c' if proba[1] > 0.5 else '#2ecc71'
            ax.barh(['Non Diabétique', 'Diabétique'],
                    [proba[0]*100, proba[1]*100],
                    color=['#2ecc71', '#e74c3c'], alpha=0.85, height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probabilité (%)', color='white')
            ax.tick_params(colors='white')
            for s in ax.spines.values(): s.set_edgecolor('#333')
            ax.text(proba[0]*100 + 1, 0, f'{proba[0]*100:.1f}%', va='center', color='white', fontweight='bold')
            ax.text(proba[1]*100 + 1, 1, f'{proba[1]*100:.1f}%', va='center', color='white', fontweight='bold')
            ax.set_title('Probabilités de prédiction', color='white', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        st.caption("⚠️ Cet outil est à titre éducatif uniquement. Consultez toujours un médecin.")

st.markdown("---")
st.markdown("🩺 **App de prédiction du Diabète** | Données NHANES | Modèles : Logistique, Random Forest, Gradient Boosting")
