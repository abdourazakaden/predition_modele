import streamlit as st
import pandas as pd
import numpy as np

# ============================================
# SIDEBAR - Barre latérale
# ============================================
st.sidebar.title("⚙️ Paramètres")
st.sidebar.markdown("---")

couleur = st.sidebar.selectbox(
    "🎨 Choisir un thème :",
    ["Bleu", "Vert", "Rouge"]
)

points = st.sidebar.slider("📊 Nombre de points :", 10, 100, 30)
st.sidebar.info(f"Thème : **{couleur}** | Points : **{points}**")

# ============================================
# TITRE PRINCIPAL
# ============================================
st.title("🚀 Streamlit - Niveau 2 : Interactivité")
st.markdown("---")

# ============================================
# ONGLETS
# ============================================
tab1, tab2, tab3 = st.tabs(["🏠 Accueil", "📊 Graphiques", "📋 Formulaire"])

# ----------------------------------------
# ONGLET 1 : COLONNES
# ----------------------------------------
with tab1:
    st.header("🏠 Mise en Page avec Colonnes")
    st.write("Les colonnes permettent d'organiser le contenu côte à côte.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📦 Colonne 1")
        st.metric(label="Ventes", value="1,234", delta="+12%")

    with col2:
        st.subheader("📦 Colonne 2")
        st.metric(label="Clients", value="567", delta="+5%")

    with col3:
        st.subheader("📦 Colonne 3")
        st.metric(label="Revenus", value="9,876 €", delta="-2%")

    st.markdown("---")

    # Radio bouton
    st.subheader("🔘 Choix avec Radio")
    choix = st.radio(
        "Quel framework préférez-vous ?",
        ["Streamlit", "Flask", "Django", "FastAPI"]
    )
    st.success(f"Vous avez choisi : **{choix}** ✅")

    # Multiselect
    st.subheader("☑️ Sélection Multiple")
    langages = st.multiselect(
        "Quels langages connaissez-vous ?",
        ["Python", "JavaScript", "Java", "C++", "SQL", "R"],
        default=["Python"]
    )
    if langages:
        st.info(f"Vous connaissez : **{', '.join(langages)}**")

# ----------------------------------------
# ONGLET 2 : GRAPHIQUES INTERACTIFS
# ----------------------------------------
with tab2:
    st.header("📊 Graphiques Interactifs")
    st.write(f"Graphique avec **{points}** points (modifiable dans la sidebar)")

    # Données aléatoires
    data = pd.DataFrame(
        np.random.randn(points, 3),
        columns=["Série A", "Série B", "Série C"]
    )

    type_graphique = st.selectbox(
        "📈 Type de graphique :",
        ["Ligne", "Barre", "Aire"]
    )

    if type_graphique == "Ligne":
        st.line_chart(data)
    elif type_graphique == "Barre":
        st.bar_chart(data)
    else:
        st.area_chart(data)

    # Afficher les données
    if st.checkbox("👁️ Voir les données brutes"):
        st.dataframe(data)

# ----------------------------------------
# ONGLET 3 : FORMULAIRE
# ----------------------------------------
with tab3:
    st.header("📋 Formulaire de Contact")
    st.write("Un formulaire regroupe tous les inputs dans un seul bloc.")

    with st.form("mon_formulaire"):
        st.subheader("✍️ Remplissez le formulaire")

        col1, col2 = st.columns(2)
        with col1:
            prenom = st.text_input("👤 Prénom")
        with col2:
            nom = st.text_input("👤 Nom")

        email = st.text_input("📧 Email")

        niveau = st.select_slider(
            "💡 Niveau en Python :",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"]
        )

        message = st.text_area("💬 Votre message :", height=100)

        # Bouton de soumission
        soumis = st.form_submit_button("📨 Envoyer")

        if soumis:
            if prenom and nom and email:
                st.success(f"✅ Merci **{prenom} {nom}** ! Votre message a été envoyé.")
                st.info(f"📧 Email : {email} | 💡 Niveau : {niveau}")
                if message:
                    st.write(f"💬 Message : *{message}*")
            else:
                st.error("❌ Veuillez remplir tous les champs obligatoires !")

st.markdown("---")
st.markdown("✅ **Félicitations ! Vous avez terminé le Niveau 2 !**")
