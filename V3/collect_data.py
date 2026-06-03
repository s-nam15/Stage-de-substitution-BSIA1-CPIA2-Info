import cv2 # OpenCV -> d'accèder à la webcam et de dessiner du texte
import mediapipe as mp # détecter les mains et extraire les 21 landmarks
import csv # Pour enregistrer les donnnées dans dataset.csv
import os # Permettre de gérer les chemins de fichiers
import math # Pour calculer la distance euclidienne

# Gestes
mapping = {
    "FINGERS_JOINED": "bout_des_doigts_joints",
    "HORNS": "cornes_avec_les_doigts",
    "MIDDLE_FINGER": "doigt_dhonneur",
    "CROSSED_FINGERS": "doigts_croises",
    "POINT_UP": "index_pointant_vers_le_haut",
    "POINT_AT_USER": "index_pointant_vers_lutilisateur",
    "CROSSED_THUMB_INDEX": "main_avec_index_et_pouce_croises",
    "POINT_RIGHT": "main_avec_index_pointant_a_droite",
    "POINT_LEFT": "main_avec_index_pointant_a_gauche",
    "POINT_DOWN": "main_avec_index_pointant_vers_le_bas",
    "POINT_UP_HAND": "main_avec_index_pointant_vers_le_haut",
    "SPREAD_HAND": "main_levee_doigts_ecartes",
    "RAISED_HAND": "main_levee",
    "PALM_DOWN": "main_paume_vers_le_bas",
    "PALM_UP": "main_paume_vers_le_haut",
    "HAND_TO_THE_RIGHT": "main_vers_la_droite",
    "HAND_TO_THE_LEFT": "main_vers_la_gauche",
    "RAISED_HANDS": "mains_levees",
    "OPEN_HANDS": "mains_ouvertes",
    "HEART_HANDS": "mains_qui_forment_un_coeur",
    "OK": "ok",
    "PALMS_TOGETHER": "paume_contre_paume_doigts_vers_le_haut",
    "FIST_RIGHT": "poing_a_droite",
    "FIST_LEFT": "poing_a_gauche",
    "FRONT_FIST": "poing_de_face",
    "RAISED_FIST": "poing_leve",
    "PINCHED_FINGERS": "pouce_et_index_rapproches",
    "THUMBS_DOWN": "pouce_vers_le_bas",
    "THUMBS_UP": "pouce_vers_le_haut",
    "VULCAN": "salut_vulcain",
    "CALL_ME": "signe_appel_telephonique_avec_les_doigts",
    "LOVE_YOU": "signe_je_taime",
    "PEACE": "v_de_la_victoire",
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Récupèrer le dossier où se trouve ce fichier (collect_data.py -> Chemin absolu -> Dossier V3)
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv") # Création du chemin du dataset dans un dossier V3


# Normalisation pour réduire les variations liées à la position de la main devant la caméra.
def normalize_hand(hand_landmarks):

    features = []

    # Récupérer les deux points de référence
    wrist = hand_landmarks.landmark[0] # poignet
    middle = hand_landmarks.landmark[12] # extrémité du majeur

    # Distance de référence en appliquant la distance euclidienne
    hand_size = math.sqrt(
        (middle.x - wrist.x) ** 2
        + (middle.y - wrist.y) ** 2
        + (middle.z - wrist.z) ** 2
    )

    """
    À l'origine, il y a 21 points sur la main, mais par souci de simplicité de calcul, 
    on suppose qu'il n'y en ait que 3 : le poignet, le bout du majeur et le bout de l'index.

    wrist (poignet, 0) : (x : 10, y : 20, z : 0)
    middle (bout du majeur, 12) : (x : 10, y : 25, z : 0)
    index (bout de l'index, 8) : (x : 12, y : 24, z : 0)

    Résultat calcul de hand_size = 5 (la valeuer standard de la taille de la main de cette personne)
    """

    # Sécurité pour éviter la division par zéro
    if hand_size == 0:
        hand_size = 1

    # Parcourir les 21 landmarks (0 ~ 20)
    for lm in hand_landmarks.landmark:

        # Pour chaque point : (Coordonnées du point actuel - Coordonnées du poignet) / Taille de la main
        x = (lm.x - wrist.x) / hand_size # (10 - 10) / 5 = 0 (exemple de poignet)
        y = (lm.y - wrist.y) / hand_size # (20 - 20) / 5 = 0 (exemple de poignet)
        z = (lm.z - wrist.z) / hand_size # (0 - 0) / 5 = 0 (exemple de poignet)

        """
        1. - wrist.x (substitution) : forcer le poignet à se déplacer vers la position (0,0,0) -> aligner à l'origine
        2. / hand_size (division) : maintenir un rapport constant, quelle que soit la variation de taille de la main -> ajuster la taille
        """

        # on ajoute les points dans la liste
        features.extend([x, y, z]) # [0, 0, 0] (Le point de référence devient 0 -> aligner à l'origine) (exemple de poignet)

    return features # 21 points * 3 coordonnées = 63 valeurs enreigstrés dans la liste

    """
    D'après le calcul : [0, 0, 0, 0, 1, 0, 0.4, 0.8, 0] pour 3 (63 valeurs de base)

    Imagine l'utilisateur approche sa main très près de la caméra ce qui a pour effet de doubler la taille de toutes les coordonnées.
    Le poignet devient (20, 40, 0) et le majeur devient (20, 50, 0) -> la taille de la main (hand_size) est calculée comme 10.
    Même si on recalcule avec ces données, le résultat reste même qu'avant en axe y par exemple 1, 0.8.
    """
    
# Récupérer tous les gestes (labels) enreigstrés dans le dictionnaire mapping
valid_labels = sorted(list(mapping.keys()))

print("\n--- GESTES DISPONIBLES ---")
print(", ".join(valid_labels))

label = input("\nEntrez le nom du geste à collecter : ").upper() # Force en majuscule

# Si utilisateur saisit la geste inconnue 
if label not in mapping:
    print("Erreur: Label non présent dans le mapping.")
    exit()

# Mediapipe (charger le modèle de détection)
mp_hands = mp.solutions.hands

# On autorise jusqu'à deux mains et 70% de confiance minimum
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Afficher visuellement les articulations et les lignes des doigts (points de repère) sur l'écran de la caméra
mp_draw = mp.solutions.drawing_utils

# Ouvrir le webcam
cap = cv2.VideoCapture(0)

# Ouvrir le fichier csv
with open(DATASET_PATH, "a", newline="") as f: # "a" = Append (on ajoute des données sans supprimer l'ancien dataset)

    writer = csv.writer(f) # Permettre d'écrire dans un fichier csv

    print(f"Collecte lancée pour : {label}") # label (nom du geste) saisit par utilisateur en ligne 112
    print("Appuyez sur 'S' pour sauvegarder")

    while True:

        # Lecture webcam
        ret, frame = cap.read() # ret = valeur booléan, frame = données pixels, cap.read() = Une image extraite (une frame) de la vidéo de la webcam connectée

        # Si il n'a pas réussi à afficher le webcam
        if not ret:
            break

        # Effet miroir = 1
        frame = cv2.flip(frame, 1)

        # Conversion RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV traite les images dans l'ordre BGR (Bleu-Vert-Rouge) mais le modèle d'IA MediaPipe les traite dans l'ordre RGB

        # Détection des mains
        result = hands.process(rgb) # MediaPipe analyse l'image, les résultats de l'analyse (coordonnées des articulations des doigts, etc.) sont stockés dans result

        landmarks_to_save = [] # Liste vide destinée à stocker les 126 coordonnées qui seront finalement enregistrées dans un fichier CSV

        # Si une seule main est détectée sur l'écran
        if result.multi_hand_landmarks:

            # Trier les mains trouvées de gauche à droite à l'écran et stocke dans hands_sorted sous forme de liste pour éviter le MediaPipe inverser main gauche et droite
            hands_sorted = sorted(
                result.multi_hand_landmarks, key=lambda h: h.landmark[0].x # landmark[0] = poignet, plus la valeur de x est petite -> situé à gauche. En utilisant lambda, on commence à trier par la valeur plus petite
            )

            # Retirer les mains alignées (hands_sorted) une par une (une fois pour une main, deux fois pour deux mains)
            for hand_landmarks in hands_sorted:

                # Dessin des landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # HAND_CONNECTIONS = relier les points pour organiser l'ossature des doigts

                # On normalise avec la fonction déjà crée en ligne 50
                normalized = normalize_hand(hand_landmarks)

                # On ajoute les données normalisés dans la liste destinée à stocker les 126 coordonnées qui seront finalement enregistrées dans un fichier CSV
                landmarks_to_save.extend(normalized)

            # Si c'est une main
            if len(hands_sorted) == 1:
                landmarks_to_save.extend([0.0] * 63) # Zero Padding (remplissage par zéros) pour chaque ligne du fichier CSV possède exactement le même nombre de valeurs

        # Affichage

        # Nombre de main(s) apparaît sur l'écran
        nb_mains = (
            len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        )

        # Afficher le nom du geste ainsi le nombre de main apparaît
        cv2.putText(
            frame, # Ecran de webcam pour écrire le nom du geste et le nombre de main
            f"Label: {label} | Mains: {nb_mains}",
            (10, 30), # x : 10 pixel, y :30 pixel -> haut à gauche
            cv2.FONT_HERSHEY_SIMPLEX, # police par défaut fourni par OpenCV
            0.7, # Rapport de taille de police
            (255, 255, 0), # Couleur de texte (BGR pour OpenCV)
            2, # épaisseur des lignes de texte
        )

        # Afficher le frame l'écran pour on puisse visualiser
        cv2.imshow("Collecte Multi-Mains", frame) # imshow -> pour afficher une image en window

        # Détection du clavier
        key = cv2.waitKey(1) & 0xFF # on attend 1 ms (= 0.001s) avant utilisateur saisit le bouton et 0xFF (hexadecimal) pour la même saisie de touches sous Windows et Linux

        # Touche S pour sauvegarder et les données sont remplies 
        if key == ord("s") and len(landmarks_to_save) == 126:

            # on enregistre vraiment dans un fichier CSV une ligne de data en contenant 126 valeurs ainsi le nom du geste
            writer.writerow(landmarks_to_save + [label])

            print("✅ Enregistré")

        # Touche ESC pour s'arrêter
        elif key == 27:
            break

# FIN
cap.release() # On ferme le webcam
cv2.destroyAllWindows() # Fermer de force toutes les fenêtres liées à OpenCV (« Collecte Multi-Mains ») qui étaient ouvertes à l'écran et effacer les complètement de l'espace mémoire de l'ordinateur.
