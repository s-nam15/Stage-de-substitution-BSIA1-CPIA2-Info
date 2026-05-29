import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# ===== CHEMIN FIXE =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")

print("Loading dataset from:", DATASET_PATH)

# charger dataset
data = pd.read_csv(DATASET_PATH, header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modèle
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# précision
print("Accuracy:", model.score(X_test, y_test))

# sauvegarde
joblib.dump(model, MODEL_PATH)
print("Model saved at:", MODEL_PATH)