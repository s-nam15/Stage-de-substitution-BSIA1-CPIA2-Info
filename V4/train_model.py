import pandas as pd # Utilisé pour lire et traiter facilement les fichiers CSV au format tableau
import numpy as np # Pour calculer les opérations mathématiques sur les matrices (Data Augmentation)
from sklearn.model_selection import train_test_split # Fonction pour répartir le data pour entraîner et le data pour tester
from sklearn.neighbors import KNeighborsClassifier # algorithme KNN
import joblib # Enregistrer le modèle machine learning entrâiné sous forme de fichier (.pkl)
import os # Permettre de gérer les chemins de fichiers

# Création du chemin
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Récupèrer le dossier où se trouve ce fichier (train_model.py -> Chemin absolu)
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv") # Création du chemin du dataset
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl") # Création du chemin du gesture_model (.pkl)

print("Loading dataset from:", DATASET_PATH)

# Charger dataset
data = pd.read_csv(DATASET_PATH, header=None) # header = None : pour éviter le pandas identifie la 1ere ligne soit le nom du colonne

# Division x (données) et y (label = correction)
X = data.iloc[:, :-1] # Extraire les données de coordonnées (les 126 valeurs)
y = data.iloc[:, -1] # Extraire le label (nom du geste) -> dernière colonne

# Fonction de rotation (sample : 126 coordonnées articulaires, angle : angle de rotation)
def rotate_sample(sample, angle):
    # On sépare le vecteur plat de 126 valeurs en 2 mains de 63 valeurs
    # Chaque main contient 21 points possédant 3 coordonnées (x, y, z)
    # On remet temporairement au format (42 points au total pour 2 mains, 3 coordonnées)
    sample_reshaped = sample.reshape(-1, 3) # -1 pour 42 lignes et 3 pour 3 colonnes

    """ Devient comme ça pour faciliter le calcul de rotation après
    [
        [X0,  Y0,  Z0],   # point 0 = poignet de première main
        [X1,  Y1,  Z1],   # point 1
        [X2,  Y2,  Z2],   # point 2
        ...
        [X41, Y41, Z41]   # point 41 = dernier point de deuxième main
    ]
    """
    
    # Valeur de trigonométriques (cos et sin)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Applique une rotation 2D uniquement sur le plan X/Y pour chaque point
    for i in range(len(sample_reshaped)):
        # Si le point est à (0.0, 0.0, 0.0), c'est du Zero Padding (main absente), on ne le tourne pas
        if np.all(sample_reshaped[i] == 0.0):
            continue
            
        x, y = sample_reshaped[i][0], sample_reshaped[i][1]

        # Formule 2D Rotation
        sample_reshaped[i][0] = x * cos_a - y * sin_a
        sample_reshaped[i][1] = x * sin_a + y * cos_a
        
    # On retransforme en vecteur plat de 126 valeurs
    return sample_reshaped.flatten()

# APPLICATION DE LA DATA AUGMENTATION (VARIÉTÉ ARTIFICIELLE)

# Liste pour stocker les nouvelles fuasses données
X_aug = []
y_aug = []

for i in range(len(X)):
    sample = X.iloc[i].values.astype(float) # les 126 données
    current_label = y.iloc[i] # le nom du geste
    
    # On conserve précieusement l'exemple original avant la rotation
    X_aug.append(sample)
    y_aug.append(current_label)
    
    # On génère 3 variantes artificielles uniques pour cet échantillon donc fois 4 par rapport les données originals pour entraîner
    for _ in range(3):
        s = sample.copy() # copie les données dans toucher les données originals
        
        # 1. Ajout de bruit gaussien (simule l'imprécision/tremblement de la caméra)
        # On n'applique pas de bruit sur les valeurs masquées par le Zero Padding (égales à 0.0)
        mask = s != 0.0
        s[mask] = s[mask] + np.random.normal(0, 0.01, np.sum(mask))
        
        # 2. Variation d’échelle (simule la distance de la main par rapport à la caméra)
        scale = np.random.uniform(0.9, 1.1)  # Variation de +/- 10%
        s = s * scale
        
        # 3. Rotation légère (simule l'inclinaison de la main de l'utilisateur)
        angle = np.random.uniform(-0.2, 0.2)  # Entre environ -11° et +11°
        s = rotate_sample(s, angle)
        
        # Ajout de la version augmentée dans nos nouvelles listes
        X_aug.append(s)
        y_aug.append(current_label)

# Remplacement des données initiales par nos données enrichies en variété
print("Applique la Data Augmentation pour ajouter de la variété...")
X = pd.DataFrame(X_aug)
y = pd.Series(y_aug)

print(f"Nombre total d'échantillons : {len(X)}")

# Entraînement du modèle KNN

# Séparer l'ensemble des données augmentées en "entraînement" (80%) et "test" (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle KNN 
model = KNeighborsClassifier(n_neighbors = 10) # On définit le nombre de voisin plus proche k = 10
model.fit(X_train, y_train) # Apprentissage du modèle avec la mine de données variées

# Précision (la note d'examen finale sur des données qu'il n'a pas vues)
accuracy = model.score(X_test, y_test)
print(f"Accuracy après injection de variété : {accuracy * 100:.2f}%")

# Sauvegarde du nouveau modèle robuste
joblib.dump(model, MODEL_PATH)
print("Model enregistré avec succès au chemin :", MODEL_PATH)