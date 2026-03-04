import streamlit as st
import pandas as pd
import numpy as np
import time

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="Streamlit Niveau 3",
    page_icon="🚀",
    layout="wide"
)

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("🚀 Niveau 3 - Avancé")
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "📌 Navigation :",
    ["📂 Charger des données", "⚡ Cache & Performance", "📈 Graphiques Avancés", "🔄 Session State", "☁️ Déploiement"]
)

# ============================================
# TITRE
# ============================================
st.title("🚀 Streamlit - Niveau 3 : Avancé")
st.markdown("---")

# ============================================
# SECTION 1 : CHARGER DES DONNÉES CSV
# ============================================
if section == "📂 Charger des données":
    st.header("📂 Charger des Données")
    st.write("Streamlit peut lire des fichiers CSV, Excel, JSON facilement.")

    st.subheader("1️⃣ Uploader un fichier CSV")
    fichier = st.file_uploader("📁 Choisir un fichier CSV", type=["csv"])

    if fichier is not None:
        df = pd.read_csv(fichier)
        st.success(f"✅ Fichier chargé ! {df.shape[0]} lignes et {df.shape[1]} colonnes")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👁️ Aperçu des données")
            st.dataframe(df.head(10))
        with col2:
            st.subheader("📊 Statistiques")
            st.dataframe(df.describe())

        st.subheader("📈 Visualiser une colonne")
        colonnes_numeriques = df.select_dtypes(include=np.number).columns.tolist()
        if colonnes_numeriques:
            col_choisie = st.selectbox("Choisir une colonne :", colonnes_numeriques)
            st.line_chart(df[col_choisie])
        else:
            st.warning("Aucune colonne numérique trouvée.")
    else:
        st.info("ℹ️ Pas de fichier ? Voici un exemple avec des données générées :")
        df_exemple = pd.DataFrame({
            "Mois"   : ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun"],
            "Ventes" : [1200, 1500, 1800, 1400, 2000, 2300],
            "Clients": [30, 45, 52, 41, 60, 75],
            "Revenus": [5000, 6200, 7800, 6100, 8900, 10200]
        })
        st.dataframe(df_exemple)
        st.line_chart(df_exemple.set_index("Mois"))

# ============================================
# SECTION 2 : CACHE & PERFORMANCE
# ============================================
elif section == "⚡ Cache & Performance":
    st.header("⚡ Cache & Performance")
    st.write("Le cache évite de recalculer les mêmes données à chaque rechargement.")

    st.subheader("🐢 Sans Cache (lent)")
    st.code("""
def charger_donnees_lent():
    time.sleep(2)  # Simule un chargement long
    return pd.DataFrame(np.random.randn(100, 3))
    """, language="python")

    st.subheader("⚡ Avec Cache (rapide)")
    st.code("""
@st.cache_data
def charger_donnees_rapide():
    time.sleep(2)  # Exécuté UNE SEULE FOIS
    return pd.DataFrame(np.random.randn(100, 3))
    """, language="python")

    st.markdown("---")
    st.subheader("🧪 Tester la différence")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🐢 Charger SANS cache"):
            with st.spinner("Chargement en cours..."):
                time.sleep(2)
            st.error("⏱️ 2 secondes à chaque clic !")

    with col2:
        @st.cache_data
        def charger_avec_cache():
            time.sleep(2)
            return pd.DataFrame(np.random.randn(50, 3), columns=["A", "B", "C"])

        if st.button("⚡ Charger AVEC cache"):
            debut = time.time()
            df = charger_avec_cache()
            fin = time.time()
            if fin - debut < 0.1:
                st.success(f"⚡ Instantané grâce au cache !")
            else:
                st.info(f"⏱️ Premier chargement : {fin-debut:.1f}s (mis en cache !)")
            st.dataframe(df.head())

    st.info("💡 **Astuce** : Utilisez `@st.cache_data` pour les données et `@st.cache_resource` pour les modèles ML.")

# ============================================
# SECTION 3 : GRAPHIQUES AVANCÉS
# ============================================
elif section == "📈 Graphiques Avancés":
    st.header("📈 Graphiques Avancés avec Plotly")
    st.write("Plotly permet des graphiques interactifs et professionnels.")

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Données exemple
        df = pd.DataFrame({
            "Mois"      : ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun", "Jul", "Aou", "Sep"],
            "Ventes"    : [1200, 1500, 1800, 1400, 2000, 2300, 2100, 2500, 2800],
            "Clients"   : [30, 45, 52, 41, 60, 75, 68, 80, 95],
            "Satisfaction": [4.2, 4.5, 4.1, 4.6, 4.8, 4.3, 4.7, 4.9, 4.5]
        })

        type_graph = st.selectbox("Choisir un type :", [
            "📈 Ligne", "📊 Barres", "🫧 Bulles", "🥧 Camembert"
        ])

        if type_graph == "📈 Ligne":
            fig = px.line(df, x="Mois", y=["Ventes", "Clients"],
                         title="Évolution des Ventes et Clients",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)

        elif type_graph == "📊 Barres":
            fig = px.bar(df, x="Mois", y="Ventes",
                        color="Ventes", title="Ventes par Mois",
                        color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

        elif type_graph == "🫧 Bulles":
            fig = px.scatter(df, x="Clients", y="Ventes",
                           size="Satisfaction", color="Mois",
                           title="Clients vs Ventes (taille = Satisfaction)",
                           hover_name="Mois")
            st.plotly_chart(fig, use_container_width=True)

        elif type_graph == "🥧 Camembert":
            fig = px.pie(df, values="Ventes", names="Mois",
                        title="Répartition des Ventes")
            st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("⚠️ Plotly non installé. Lancez : `pip install plotly`")
        st.subheader("Graphiques natifs Streamlit :")
        df = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
        st.area_chart(df)

# ============================================
# SECTION 4 : SESSION STATE
# ============================================
elif section == "🔄 Session State":
    st.header("🔄 Session State")
    st.write("Le Session State permet de garder des données en mémoire pendant la session.")

    st.subheader("🛒 Exemple : Mini Panier d'Achats")

    # Initialiser le panier
    if "panier" not in st.session_state:
        st.session_state.panier = []
    if "total" not in st.session_state:
        st.session_state.total = 0.0

    produits = {
        "🍎 Pomme"    : 0.50,
        "🍌 Banane"   : 0.30,
        "🥛 Lait"     : 1.20,
        "🍞 Pain"     : 2.50,
        "🧀 Fromage"  : 3.80,
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏪 Produits disponibles")
        for produit, prix in produits.items():
            if st.button(f"Ajouter {produit} ({prix}€)"):
                st.session_state.panier.append(produit)
                st.session_state.total += prix

    with col2:
        st.subheader("🛒 Votre Panier")
        if st.session_state.panier:
            for item in st.session_state.panier:
                st.write(f"• {item}")
            st.markdown("---")
            st.success(f"💰 Total : **{st.session_state.total:.2f} €**")
            if st.button("🗑️ Vider le panier"):
                st.session_state.panier = []
                st.session_state.total = 0.0
                st.rerun()
        else:
            st.info("Votre panier est vide.")

    st.markdown("---")
    st.subheader("💡 Comment ça marche ?")
    st.code("""
# Initialiser une variable
if "compteur" not in st.session_state:
    st.session_state.compteur = 0

# Modifier la variable
if st.button("Incrémenter"):
    st.session_state.compteur += 1

# Afficher la variable
st.write(f"Compteur : {st.session_state.compteur}")
    """, language="python")

# ============================================
# SECTION 5 : DÉPLOIEMENT
# ============================================
elif section == "☁️ Déploiement":
    st.header("☁️ Déployer votre App en ligne")
    st.write("Streamlit Cloud permet de déployer gratuitement en quelques minutes !")

    st.subheader("📋 Étapes pour déployer :")

    etapes = [
        ("1️⃣", "Créer un compte GitHub", "https://github.com", "Gratuit"),
        ("2️⃣", "Uploader votre code sur GitHub", "Créer un nouveau repository", ""),
        ("3️⃣", "Créer un compte Streamlit Cloud", "https://share.streamlit.io", "Gratuit"),
        ("4️⃣", "Connecter votre repo GitHub", "Cliquer sur 'New app'", ""),
        ("5️⃣", "Déployer !", "Votre app est en ligne en 2 minutes", "🚀"),
    ]

    for numero, titre, detail, badge in etapes:
        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            st.subheader(numero)
        with col2:
            st.write(f"**{titre}**")
            st.caption(detail)
        with col3:
            if badge:
                st.success(badge)
        st.markdown("---")

    st.subheader("📁 Structure de votre projet :")
    st.code("""
mon_projet/
│
├── app.py              ← Votre application Streamlit
├── requirements.txt    ← Les bibliothèques nécessaires
└── README.md           ← Description du projet
    """)

    st.subheader("📄 Exemple de requirements.txt :")
    st.code("""
streamlit
pandas
numpy
plotly
    """)

    st.success("🎉 Félicitations ! Vous maîtrisez maintenant Streamlit du Niveau 1 au Niveau 3 !")
    st.balloons()

st.markdown("---")
st.markdown("🏆 **Vous êtes maintenant un développeur Streamlit !**")
