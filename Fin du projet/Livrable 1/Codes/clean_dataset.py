import os

# ===== CONFIGURATION =====
# Définition des chemins d'accès aux fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

# Nom du geste à réinitialiser (supprimer)
TARGET_GESTURE = ""

if os.path.exists(DATASET_PATH):
    # 1. Lecture des données existantes
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 2. Filtrage : on garde uniquement les lignes qui ne contiennent PAS le geste cible
    clean_lines = [line for line in lines if TARGET_GESTURE not in line]
    
    # 3. Réécriture du fichier mis à jour (écrasement sécurisé)
    with open(DATASET_PATH, "w", encoding="utf-8", newline="") as f:
        f.writelines(clean_lines)
        
    # Calcul du nombre de lignes supprimées
    deleted_count = len(lines) - len(clean_lines)
    print(f"✅ Suppression réussie ! Au total, {deleted_count} lignes de données '{TARGET_GESTURE}' ont été supprimées.")
else:
    print("❌ Erreur : Le fichier 'dataset.csv' est introuvable.")