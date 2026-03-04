import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
print("=" * 50)
print("  ÉTAPE 1 : Chargement des données")
print("=" * 50)

df = pd.read_csv(r'C:\Users\hp\Downloads\diabete1.csv', sep=';')
print(f"✅ Dataset chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"\nColonnes : {df.columns.tolist()}")
print(f"\nAperçu :\n{df.head()}")

# ============================================================
# 2. EXPLORATION RAPIDE
# ============================================================
print("\n" + "=" * 50)
print("  ÉTAPE 2 : Exploration des données")
print("=" * 50)

print(f"\n📊 Distribution de la cible (Outcome) :")
print(df['Outcome'].value_counts())
print(f"\n  ➜ Non diabétiques (0) : {(df['Outcome']==0).sum()} patients")
print(f"  ➜ Diabétiques     (1) : {(df['Outcome']==1).sum()} patients")
print(f"  ➜ Taux de diabète     : {df['Outcome'].mean()*100:.1f}%")

print(f"\n📋 Statistiques descriptives :")
print(df.describe().round(2))

# ============================================================
# 3. NETTOYAGE DES DONNÉES
# ============================================================
print("\n" + "=" * 50)
print("  ÉTAPE 3 : Nettoyage des données")
print("=" * 50)

# Les colonnes qui ne peuvent pas valoir 0 biologiquement
cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\n⚠️  Valeurs à 0 impossibles biologiquement (remplacées par NaN) :")
for col in cols_zero:
    nb_zeros = (df[col] == 0).sum()
    print(f"  ➜ {col:25s} : {nb_zeros} zéros")

df[cols_zero] = df[cols_zero].replace(0, np.nan)

print(f"\n🔍 Valeurs manquantes après nettoyage :")
print(df.isnull().sum())

# ============================================================
# 4. SÉPARATION FEATURES / TARGET
# ============================================================
print("\n" + "=" * 50)
print("  ÉTAPE 4 : Préparation des données")
print("=" * 50)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = df[features]
y = df['Outcome']

print(f"\n✅ Features utilisées ({len(features)}) : {features}")
print(f"✅ Variable cible    : Outcome (0=Non Diabétique, 1=Diabétique)")

# ============================================================
# 5. SPLIT TRAIN / TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% train, 20% test
    random_state=42,     # reproductibilité
    stratify=y           # garder la proportion diabète/non-diabète
)

print(f"\n📦 Séparation Train/Test :")
print(f"  ➜ Train : {X_train.shape[0]} patients ({X_train.shape[0]/len(df)*100:.0f}%)")
print(f"  ➜ Test  : {X_test.shape[0]}  patients ({X_test.shape[0]/len(df)*100:.0f}%)")

# ============================================================
# 6. IMPUTATION DES VALEURS MANQUANTES
# ============================================================
print(f"\n🔧 Imputation des valeurs manquantes (médiane)...")
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)  # fit sur train uniquement !
X_test_imp  = imputer.transform(X_test)       # transform seulement sur test

print(f"✅ Imputation terminée")

# ============================================================
# 7. NORMALISATION
# ============================================================
print(f"\n🔧 Normalisation (StandardScaler)...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)  # fit sur train uniquement !
X_test_sc  = scaler.transform(X_test_imp)       # transform seulement sur test

print(f"✅ Normalisation terminée")

# ============================================================
# 8. ENTRAÎNEMENT DES MODÈLES
# ============================================================
print("\n" + "=" * 50)
print("  ÉTAPE 5 : Entraînement des modèles")
print("=" * 50)

models = {
    'Régression Logistique': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ),
}

resultats = {}

for nom, model in models.items():
    print(f"\n⚙️  Entraînement : {nom}...")

    # Entraînement
    model.fit(X_train_sc, y_train)

    # Prédictions
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    # Scores
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Cross-validation (5 folds)
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='roc_auc')

    resultats[nom] = {
        'model'  : model,
        'acc'    : acc,
        'auc'    : auc,
        'cv_mean': cv_scores.mean(),
        'cv_std' : cv_scores.std(),
        'y_pred' : y_pred,
        'y_prob' : y_prob,
    }

    print(f"  ✅ Accuracy        : {acc*100:.2f}%")
    print(f"  ✅ AUC Score       : {auc:.4f}")
    print(f"  ✅ CV AUC (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 9. COMPARAISON DES MODÈLES
# ============================================================
print("\n" + "=" * 50)
print("  ÉTAPE 6 : Comparaison des modèles")
print("=" * 50)

print(f"\n{'Modèle':<30} {'Accuracy':>10} {'AUC':>10} {'CV AUC':>15}")
print("-" * 70)
for nom, res in resultats.items():
    print(f"{nom:<30} {res['acc']*100:>9.2f}% {res['auc']:>10.4f} "
          f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}")

# Meilleur modèle
best_name = max(resultats, key=lambda x: resultats[x]['auc'])
best_res   = resultats[best_name]
print(f"\n🥇 MEILLEUR MODÈLE : {best_name}")
print(f"   AUC = {best_res['auc']:.4f} | Accuracy = {best_res['acc']*100:.2f}%")

# ============================================================
# 10. RAPPORT DÉTAILLÉ DU MEILLEUR MODÈLE
# ============================================================
print("\n" + "=" * 50)
print(f"  ÉTAPE 7 : Rapport détaillé — {best_name}")
print("=" * 50)

print(f"\n📋 Rapport de classification :")
print(classification_report(
    y_test, best_res['y_pred'],
    target_names=['Non Diabétique', 'Diabétique']
))

print(f"\n🔲 Matrice de confusion :")
cm = confusion_matrix(y_test, best_res['y_pred'])
print(f"\n                  Prédit Non    Prédit Oui")
print(f"  Réel Non      :    {cm[0,0]:5d}        {cm[0,1]:5d}")
print(f"  Réel Oui      :    {cm[1,0]:5d}        {cm[1,1]:5d}")

tn, fp, fn, tp = cm.ravel()
print(f"\n  ➜ Vrais Négatifs  (TN) : {tn}  — Non diabétiques correctement détectés")
print(f"  ➜ Vrais Positifs  (TP) : {tp}  — Diabétiques correctement détectés")
print(f"  ➜ Faux Positifs   (FP) : {fp}   — Non diabétiques classés diabétiques")
print(f"  ➜ Faux Négatifs   (FN) : {fn}  — Diabétiques non détectés ⚠️")

# ============================================================
# 11. IMPORTANCE DES VARIABLES
# ============================================================
print("\n" + "=" * 50)
print("  ÉTAPE 8 : Importance des variables (Random Forest)")
print("=" * 50)

rf = models['Random Forest']
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

labels_fr = {
    'Glucose'                  : 'Glucose',
    'BMI'                      : 'IMC (BMI)',
    'Age'                      : 'Âge',
    'DiabetesPedigreeFunction' : 'Pedigree Diabète',
    'Insulin'                  : 'Insuline',
    'BloodPressure'            : 'Tension artérielle',
    'SkinThickness'            : 'Épaisseur peau',
    'Pregnancies'              : 'Grossesses',
}

print(f"\n{'Variable':<30} {'Importance':>12}")
print("-" * 45)
for feat, val in imp.items():
    bar = '█' * int(val * 100)
    print(f"  {labels_fr.get(feat, feat):<28} {val:.4f}  {bar}")

# ============================================================
# 12. SAUVEGARDE DU MODÈLE
# ============================================================
print("\n" + "=" * 50)
print("  ÉTAPE 9 : Sauvegarde du modèle")
print("=" * 50)

sauvegarde = {
    'models'  : models,
    'imputer' : imputer,
    'scaler'  : scaler,
    'features': features,
    'best'    : best_name,
}

with open('modele_diabete.pkl', 'wb') as f:
    pickle.dump(sauvegarde, f)

print(f"\n✅ Modèle sauvegardé : modele_diabete.pkl")
print(f"   Ce fichier contient :")
print(f"   ➜ Les 3 modèles entraînés")
print(f"   ➜ L'imputer (valeurs manquantes)")
print(f"   ➜ Le scaler (normalisation)")
print(f"   ➜ La liste des features")

# ============================================================
# 13. EXEMPLE DE PRÉDICTION
# ============================================================
print("\n" + "=" * 50)
print("  ÉTAPE 10 : Exemple de prédiction")
print("=" * 50)

# Patient exemple
patient_exemple = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]],
                                columns=features)

print(f"\n👤 Patient exemple :")
for feat, val in zip(features, patient_exemple.values[0]):
    print(f"   ➜ {labels_fr.get(feat, feat):<28} : {val}")

p_imp = imputer.transform(patient_exemple)
p_sc  = scaler.transform(p_imp)

best_model = models[best_name]
pred  = best_model.predict(p_sc)[0]
proba = best_model.predict_proba(p_sc)[0]

print(f"\n🔮 Résultat ({best_name}) :")
print(f"   ➜ Prédiction       : {'⚠️  DIABÉTIQUE' if pred == 1 else '✅ NON DIABÉTIQUE'}")
print(f"   ➜ Proba Non Diab.  : {proba[0]*100:.1f}%")
print(f"   ➜ Proba Diabète    : {proba[1]*100:.1f}%")

print("\n" + "=" * 50)
print("  ✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 50)
print("\n📁 Fichier généré : modele_diabete.pkl")
print("🚀 Vous pouvez maintenant lancer : streamlit run app_diabete_final.py\n")
