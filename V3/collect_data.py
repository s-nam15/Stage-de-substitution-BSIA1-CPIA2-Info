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


# ===== NORMALISATION =====
def normalize_hand(hand_landmarks):

    features = []

    wrist = hand_landmarks.landmark[0]
    middle = hand_landmarks.landmark[12]

    # Distance de référence
    hand_size = math.sqrt(
        (middle.x - wrist.x) ** 2
        + (middle.y - wrist.y) ** 2
        + (middle.z - wrist.z) ** 2
    )

    # Sécurité
    if hand_size == 0:
        hand_size = 1

    # Normalisation
    for lm in hand_landmarks.landmark:

        x = (lm.x - wrist.x) / hand_size
        y = (lm.y - wrist.y) / hand_size
        z = (lm.z - wrist.z) / hand_size

        features.extend([x, y, z])

    return features


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
