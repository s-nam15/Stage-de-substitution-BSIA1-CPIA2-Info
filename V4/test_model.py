import cv2 # OpenCV -> d'accèder à la webcam et de dessiner du texte
import mediapipe as mp # détecter les mains et extraire les 21 landmarks
import joblib # Enregistrer le modèle machine learning entrâiné sous forme de fichier (.pkl)
import os # Permettre de gérer les chemins de fichiers
import numpy as np # Pour calculer l'algèbre linéaire
import math # Pour calculer la distance euclidienne
import pyautogui
import time


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

# 70% de confiance des gestes
CONFIDENCE_THRESHOLD = 0.7

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Récupèrer le dossier où se trouve ce fichier (test_model.py -> Chemin absolu -> Dossier V3)

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl") # Création du chemin du gesture_model (modèle machine learning sous forme de fichier) dans un dossier V3
IMG_DIR = os.path.join(BASE_DIR, "static", "img") # Création du chemin du dossier static -> dossier img dans un dossier V3

# Contrôle de la vidéo
last_command_time = 0
COMMAND_DELAY = 2  # secondes -> Délai d'attente pour éviter l'exécution consécutive de la même commande
music_paused = False # etat initial = musique en train de relancer, True si la musique en pause


# Chargement du modèle
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé.")
else:
    print(f"❌ Modèle introuvable à : {MODEL_PATH}")
    exit()

# Chargement des images depuis dans un dossier img
gesture_images = {} # Dictionnaire pour stocker les images

# Boucle pour récupérer le label et le nom d'image dans un dictionnaire mapping
for ml_label, file_name in mapping.items():
    # Création du chemin où l'image est stockée
    path = os.path.join(IMG_DIR, f"{file_name}.png")

    # Récupérer l'image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Si il a réussi à récupérer l'image
    if img is not None:
        # on les stocke dans un dictionnaire qu'on a crée à la ligne 67
        gesture_images[ml_label] = img


# Normalisation pour réduire les variations liées à la position de la main devant la caméra.
def normalize_hand(hand_landmarks):
    features = []

    # Récupérer les deux points de référence
    wrist = hand_landmarks.landmark[0] # poignet
    middle = hand_landmarks.landmark[12] # # extrémité du majeur

    # Distance de référence en appliquant la distance euclidienne (= taille de main)
    hand_size = math.sqrt(
        (middle.x - wrist.x) ** 2
        + (middle.y - wrist.y) ** 2
        + (middle.z - wrist.z) ** 2
    )

    # Sécurité pour éviter la division par zéro
    if hand_size == 0:
        hand_size = 1

    # Parcourir les 21 landmarks (0 ~ 20)
    for lm in hand_landmarks.landmark:
        # Pour chaque point : (Coordonnées du point actuel - Coordonnées du poignet) / Taille de la main
        x = (lm.x - wrist.x) / hand_size 
        y = (lm.y - wrist.y) / hand_size 
        z = (lm.z - wrist.z) / hand_size 

        # on ajoute les points dans la liste
        features.extend([x, y, z]) 

    return features 


# Fonction Overlay PNG transparent pour afficher plus dynamiquement
def overlay_emoji_transparent(frame, emoji_img, x, y, size=120): 
    if emoji_img is None:
        return

    try:
        emoji_res = cv2.resize(emoji_img, (size, size), interpolation=cv2.INTER_AREA)
        h_f, w_f, _ = frame.shape 

        y1 = max(0, y)
        y2 = min(h_f, y + size)
        x1 = max(0, x)
        x2 = min(w_f, x + size)

        img_y1 = 0 + (y1 - y)
        img_y2 = size - (y + size - y2)
        img_x1 = 0 + (x1 - x)
        img_x2 = size - (x + size - x2)

        if (y2 - y1) <= 0 or (x2 - x1) <= 0:
            return

        crop_emoji = emoji_res[img_y1:img_y2, img_x1:img_x2]
        crop_frame = frame[y1:y2, x1:x2]

        if crop_emoji.shape[2] == 4:
            alpha = crop_emoji[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            rgb_emoji = crop_emoji[:, :, :3]
            blended = rgb_emoji * alpha + crop_frame * (1.0 - alpha)
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = crop_emoji
    except Exception:
        pass


# Variables pour lissage d'émoji
smooth_x, smooth_y = 0, 0
is_first_frame = True

# Mediapipe (charger le modèle de détection)
mp_hands = mp.solutions.hands

# On autorise jusqu'à deux mains et 70% de confiance minimum
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Afficher visuellement les articulations et les lignes des doigts sur l'écran
mp_draw = mp.solutions.drawing_utils

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 

    if not ret:
        break

    # Effet miroir
    frame = cv2.flip(frame, 1)
    h_f, w_f, _ = frame.shape 

    # Conversion RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des mains
    result = hands.process(rgb) 

    if result.multi_hand_landmarks:
        test_features = [] 

        # Trier les mains de gauche à droite
        hands_sorted = sorted(
            result.multi_hand_landmarks, key=lambda h: h.landmark[0].x 
        )

        for hl in hands_sorted:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS) 
            normalized = normalize_hand(hl)
            test_features.extend(normalized)

        # Si c'est une seule main, on applique le Zero Padding
        if len(hands_sorted) == 1:
            test_features.extend([0.0] * 63) 

        # Calculer la probabilité de prédiction
        if len(test_features) == 126: 
            try:
                # Obtenir la probabilité de chaque geste depuis le modèle
                probabilities = model.predict_proba([test_features])[0] 

                # Obtenir la probabilité max et conversion en pourcentage entier
                max_prob = np.max(probabilities)
                percentage = int(max_prob * 100)

                if max_prob >= CONFIDENCE_THRESHOLD: 
                    class_idx = np.argmax(probabilities) 
                    pred_key = model.classes_[class_idx]

                    # Manipulation de la vidéo en réalisant les gestes (poing levee, main levee doigts ecartes, main vers la gauche et la droite)
                    
                    # Heure actuelle en secondes
                    current_time = time.time()

                    # Vérifier si plus de 2s s'est passé depuis la dernière commande
                    if current_time - last_command_time > COMMAND_DELAY:

                        # Pause (poing levee)
                        if pred_key == "RAISED_FIST" and not music_paused:
                             pyautogui.press("playpause") # touche l'appui play et pause
                             music_paused = True
                             last_command_time = current_time # heure actuelle devient comme l'heure de dernière commande effecuté 
                             print("⏸ Musique en pause")


                        # Reprise (main levee doigts ecartes)
                        elif pred_key == "SPREAD_HAND" and music_paused:
                             pyautogui.press("playpause") 
                             music_paused = False
                             last_command_time = current_time
                             print("▶️ Musique relancée")


                        # Suivant dans la playlist (main vers la droite)
                        elif pred_key == "HAND_TO_THE_RIGHT":
                            pyautogui.press("nexttrack") # suivant
                            last_command_time = current_time
                            print("⏭ Musique suivante")


                        # Precedent dans la playlist (main vers la gauche)
                        elif pred_key == "HAND_TO_THE_LEFT":
                            pyautogui.press("prevtrack") # precedent
                            last_command_time = current_time
                            print("⏮ Musique précédente")

                    # Fin de manipulation de la vidéo

                    if pred_key in mapping:
                        clean_name = mapping[pred_key].replace("_", " ").capitalize()
                        pred_text = f"{clean_name} ({percentage}%)"
                    else:
                        pred_text = f"{pred_key.replace('_', ' ').capitalize()} ({percentage}%)"

                    text_color = (0, 255, 0) # Vert pour un geste validé

                # Geste inconnu (Inférieur à 70%)
                else:
                    pred_key = "INCONNU"
                    
                    # On cherche quand même le geste le plus proche pour l'afficher à titre indicatif
                    class_idx = np.argmax(probabilities)
                    potential_key = model.classes_[class_idx]
                    
                    if potential_key in mapping:
                        clean_name = mapping[potential_key].replace("_", " ").capitalize()
                    else:
                        clean_name = potential_key
                        
                    pred_text = f"Inconnu... ({clean_name} ? {percentage}%)"
                    text_color = (0, 0, 255) # Rouge pour l'incertitude

                # Positionnement sur le bout de l'index (Landmark 8)
                target_x = int(hands_sorted[0].landmark[8].x * w_f)
                target_y = int(hands_sorted[0].landmark[8].y * h_f)

                # Lissage des coordonnées
                if is_first_frame:
                    smooth_x = target_x
                    smooth_y = target_y
                    is_first_frame = False
                else:
                    smooth_x = int(smooth_x + 0.25 * (target_x - smooth_x))
                    smooth_y = int(smooth_y + 0.25 * (target_y - smooth_y))

                # Positions de l'UI
                text_y = smooth_y - 40
                emoji_y = smooth_y - 180
                emoji_x = smooth_x - 60

                # Contour noir épais pour la lisibilité
                cv2.putText(
                    frame, pred_text, (smooth_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA,
                )

                # Affichage du texte principal avec sa couleur (Vert ou Rouge)
                cv2.putText(
                    frame, pred_text, (smooth_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color, 2, cv2.LINE_AA,
                )

                # Affichage de l'émoji si le geste est identifié et possède une image
                if pred_key in gesture_images: 
                    overlay_emoji_transparent(
                        frame, gesture_images[pred_key], emoji_x, emoji_y, size=120
                    )

            except Exception as e:
                print("Erreur :", e)

    else:
        is_first_frame = True

    # Affichage de la fenêtre principale
    cv2.imshow("TIAGO Robot Recognition", frame)

    # Touche ESC pour s'arrêter
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Fermeture proprement
cap.release() 
cv2.destroyAllWindows()