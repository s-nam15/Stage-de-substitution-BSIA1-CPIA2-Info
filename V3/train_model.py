import pandas as pd # Utilisé pour lire et traiter facilement les fichiers CSV au format tableau
from sklearn.model_selection import train_test_split # Fonction pour répartir le data pour entraîner et le data pour tester
from sklearn.neighbors import KNeighborsClassifier # algorithme KNN
import joblib # Enregistrer le modèle machine learning entrâiné sous forme de fichier (.pkl)
import os # Permettre de gérer les chemins de fichiers

# Création du chemin
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Récupèrer le dossier où se trouve ce fichier (train_model.py -> Chemin absolu -> Dossier V3)
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv") # Création du chemin du dataset dans un dossier V3
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl") # Création du chemin du gesture_model (modèle machine learning sous forme de fichier) dans un dossier V3

print("Loading dataset from:", DATASET_PATH)

# Charger dataset
data = pd.read_csv(DATASET_PATH, header=None) # header = None : pour éviter le pandas identifie la 1ere ligne soit le nom du colonne

# Division x (données) et y (label = correction) pour la machine learning puisse les apprendre à part
X = data.iloc[:, :-1] # Extraire les données juste avant le tour dernier (sauf le label)
y = data.iloc[:, -1] # Extraire le label (nom du geste) -> dernier

# Séparer l'ensemble des données en "entraînement" et "test" (80% pour entraîner et 20% pour tester)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""
X_train: Sujet d'étude (80% du total X)
X_test: Sujet d'examen (20% du total X)
y_train: Corrigé du sujet détude (80% du total y)
y_test: Corrigé d'examen (20% du total y)

C'est le façon pour éviter le Surapprentissage en machine learning
"""

# Création du modèle KNN
model = KNeighborsClassifier(n_neighbors=3) # On définit le nombre de voisin plus proche k = 3
model.fit(X_train, y_train) # fit : apprentissage du modèle

# Précision (la note d'examen)
print("Accuracy:", model.score(X_test, y_test)) # score: fonction pour évaluer la performance du model entraîné

# Sauvegarde
joblib.dump(model, MODEL_PATH) # model (notre modèle KNN entraîné) enregistre dans le chemin que on a défini à la ligne 10
print("Model saved at:", MODEL_PATH) 