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
    Même si on recalcule avec ces données, le résultat reste même qu'avant en axe y par exemple (1, 0.8).
    """
    

# ===== SÉLECTION LABEL =====
valid_labels = sorted(list(mapping.keys()))

print("\n--- GESTES DISPONIBLES ---")
print(", ".join(valid_labels))

label = input("\nEntrez le nom du geste à collecter : ").upper()

if label not in mapping:
    print("Erreur: Label non présent dans le mapping.")
    exit()

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# ===== WEBCAM =====
cap = cv2.VideoCapture(0)

# ===== CSV =====
with open(DATASET_PATH, "a", newline="") as f:

    writer = csv.writer(f)

    print(f"🚀 Collecte lancée pour : {label}")
    print("Appuyez sur 'S' pour sauvegarder")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        landmarks_to_save = []

        if result.multi_hand_landmarks:

            # ===== TRI GAUCHE -> DROITE =====
            hands_sorted = sorted(
                result.multi_hand_landmarks, key=lambda h: h.landmark[0].x
            )

            for hand_landmarks in hands_sorted:

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ===== NORMALISATION =====
                normalized = normalize_hand(hand_landmarks)

                landmarks_to_save.extend(normalized)

            # ===== SI UNE MAIN =====
            if len(hands_sorted) == 1:
                landmarks_to_save.extend([0.0] * 63)

        # ===== AFFICHAGE =====
        nb_mains = (
            len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        )

        cv2.putText(
            frame,
            f"Label: {label} | Mains: {nb_mains}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Collecte Multi-Mains", frame)

        # ===== CLAVIER =====
        key = cv2.waitKey(1) & 0xFF

        # ===== SAUVEGARDE =====
        if key == ord("s") and len(landmarks_to_save) == 126:

            writer.writerow(landmarks_to_save + [label])

            print("✅ Enregistré")

        # ===== ESC =====
        elif key == 27:
            break

# ===== FIN =====
cap.release()
cv2.destroyAllWindows()
