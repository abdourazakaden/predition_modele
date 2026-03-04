import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
