import os

# ===== CONFIGURATION =====
# Définition des chemins d'accès aux fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

if os.path.exists(DATASET_PATH):
    # 1. Lecture des données existantes
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"📊 Le fichier 'dataset.csv' contient un total de {total_lines} lignes.\n")
    
    # 2. Demander à l'utilisateur ce qu'il veut faire
    print("--- OPTIONS ---")
    print("1. Compter les données d'un geste précis")
    print("2. Voir le résumé de TOUS les gestes enregistrés")
    choix = input("\nEntrez votre choix (1 ou 2) : ").strip()
    
    if choix == "1":
        # Compter pour un geste spécifique
        target_gesture = input("Entrez le nom du geste à chercher : ").strip()
        
        # Filtrage : on compte les lignes qui contiennent le geste cible
        gesture_count = sum(1 for line in lines if target_gesture in line and line.strip())
        
        print(f"\n🔍 Résultat : Le geste '{target_gesture}' possède {gesture_count} lignes de données.")
        
    elif choix == "2":
        # Résumé global automatique
        print("\n📋 Répartition des données dans le dataset :")
        gestes_trouves = {}
        
        for line in lines:
            if line.strip(): # Éviter les lignes vides
                elements = line.strip().split(",")
                label = elements[-1] # Le label est toujours le tout dernier élément
                gestes_trouves[label] = gestes_trouves.get(label, 0) + 1
        
        # Affichage trié par ordre alphabétique des gestes
        for geste, count in sorted(gestes_trouves.items()):
            print(f"  • {geste} : {count} lignes")
            
        print("\n" + "="*40)
        print("📈 STATISTIQUES GLOBALES :")
        print("="*40)
        
        # Récupération du nombre de gestes détectés
        nb_gestes = len(gestes_trouves)
        print(f"• Nombre de gestes détectés : {nb_gestes}")
        
        # Calcul de la moyenne de lignes par geste (sécurité si le fichier est vide)
        if nb_gestes > 0:
            moyenne_actuelle = total_lines / nb_gestes
            print(f"• Moyenne actuelle : {moyenne_actuelle:.1f} lignes par geste")
        else:
            print("• Moyenne actuelle : 0 lignes par geste")
            
    else:
        print("❌ Choix invalide. Fin du script.")

else:
    print("❌ Erreur : Le fichier 'dataset.csv' est introuvable.")