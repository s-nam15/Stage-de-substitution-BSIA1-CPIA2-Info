import cv2 # OpenCV -> d'accèder à la webcam et de dessiner du texte
import mediapipe as mp # détecter les mains et extraire les 21 landmarks
import joblib # Enregistrer le modèle machine learning entrâiné sous forme de fichier (.pkl)
import os # Permettre de gérer les chemins de fichiers
import numpy as np # Pour calculer l'algèbre linéaire
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

# 70% de confiance des gestes
CONFIDENCE_THRESHOLD = 0.7

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Récupèrer le dossier où se trouve ce fichier (test_model.py -> Chemin absolu -> Dossier V3)

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl") # Création du chemin du gesture_model (modèle machine learning sous forme de fichier) dans un dossier V3
IMG_DIR = os.path.join(BASE_DIR, "static", "img") # Création du chemin du dossier static -> dossier img dans un dossier V3


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

# Fonction Overlay PNG transparent pour afficher plus dynamiquement
def overlay_emoji_transparent(frame, emoji_img, x, y, size=120): 

    # Si l'image n'est pas chargée
    if emoji_img is None:
        return

    try:
        # Taille d'émoji en 120x120 et INTER_AREA pour minimiser la dégradation de l'image lors du redimensionnement
        emoji_res = cv2.resize(emoji_img, (size, size), interpolation=cv2.INTER_AREA)

        # Récupérer les dimensions verticales (h_f) et horizontales (w_f) de l'écran
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

        # PNG transparent
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


# Variblels pour lissage d'émoji
smooth_x, smooth_y = 0, 0
is_first_frame = True

# Mediapipe (charger le modèle de détection)
mp_hands = mp.solutions.hands

# On autorise jusqu'à deux mains et 70% de confiance minimum
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Afficher visuellement les articulations et les lignes des doigts (points de repère) sur l'écran de la caméra
mp_draw = mp.solutions.drawing_utils

# Ouvrir le webcam
cap = cv2.VideoCapture(0)

while True:

    # Lecture webcam
    ret, frame = cap.read() # ret = valeur booléan, frame = données pixels, cap.read() = Une image extraite (une frame) de la vidéo de la webcam connectée

    # Si il n'a pas réussi à afficher le webcam
    if not ret:
        break

    # Effet miroir = 1
    frame = cv2.flip(frame, 1)

    # Récupérer les dimensions verticales (h_f) et horizontales (w_f) de l'écran
    h_f, w_f, _ = frame.shape 

    # Conversion RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV traite les images dans l'ordre BGR (Bleu-Vert-Rouge) mais le modèle d'IA MediaPipe les traite dans l'ordre RGB

    # Détection des mains
    result = hands.process(rgb) # MediaPipe analyse l'image, les résultats de l'analyse (coordonnées des articulations des doigts, etc.) sont stockés dans result

    # Si une seule main est détectée sur l'écran
    if result.multi_hand_landmarks:

        test_features = [] # Liste vide destinée à stocker les 126 coordonnées en temps réel

        # Trier les mains trouvées de gauche à droite à l'écran et stocke dans hands_sorted sous forme de liste pour éviter le MediaPipe inverser main gauche et droite
        hands_sorted = sorted(
            result.multi_hand_landmarks, key=lambda h: h.landmark[0].x # landmark[0] = poignet, plus la valeur de x est petite -> situé à gauche. En utilisant lambda, on commence à trier par la valeur plus petite
        )

        # Retirer les mains alignées (hands_sorted) une par une (une fois pour une main, deux fois pour deux mains)
        for hl in hands_sorted:

            # Dessin des landmarks
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS) # HAND_CONNECTIONS = relier les points pour organiser l'ossature des doigts

            # On normalise avec la fonction déjà crée en ligne 85
            normalized = normalize_hand(hl)

            # On ajoute les données normalisés dans la liste destinée à stocker les 126 coordonnées en temps réel
            test_features.extend(normalized)

        # Si c'est une main
        if len(hands_sorted) == 1:
            test_features.extend([0.0] * 63) # # Zero Padding (remplissage par zéros)

        # Calculer la probabilité de prédiction
        if len(test_features) == 126: # Si le nombre de données est 126 valeurs

            try:
                # Obtenir la probabilité de chaque geste depuis le modèle entraîné en utilisant les données dans test_features 
                probabilities = model.predict_proba([test_features])[0] # [0] pour éviter par exemple double crochet [[]]

                # Obtenir la probabilité max
                max_prob = np.max(probabilities)

                # Geste connu
                if max_prob >= CONFIDENCE_THRESHOLD: # Si max proba est supérieur ou égal à 70%

                    # np.argmax : trouver l'indice où le valeur la plus élevée
                    class_idx = np.argmax(probabilities) 
                    
                    # on récupère le nom du geste (label) depuis le modèle grâce à l'indice
                    pred_key = model.classes_[class_idx]

                    # Si le nom du geste est dans un dictionnaire mapping
                    if pred_key in mapping:
                        # afficher le label dynamiquement (capitalize -> Majuscule en 1er mot)
                        pred_text = mapping[pred_key].replace("_", " ").capitalize()

                    #else:
                        #pred_text = pred_key.replace("_", " ").capitalize()

                    # BGR
                    text_color = (0, 255, 0)

                # Geste inconnu
                else:

                    pred_key = "INCONNU"

                    pred_text = "Geste inconnu"

                    text_color = (0, 0, 255)

                # Position
                target_x = int(hands_sorted[0].landmark[8].x * w_f) # le bout de l'index * dimension horizontale définie à la ligne 224

                target_y = int(hands_sorted[0].landmark[8].y * h_f) # le bout de l'index * dimension verticale définie à la ligne 224

                # Lissage
                if is_first_frame:

                    smooth_x = target_x
                    smooth_y = target_y

                    is_first_frame = False

                else:

                    smooth_x = int(smooth_x + 0.25 * (target_x - smooth_x))

                    smooth_y = int(smooth_y + 0.25 * (target_y - smooth_y))

                # Positions UI
                text_y = smooth_y - 40

                emoji_y = smooth_y - 180

                emoji_x = smooth_x - 60

                # Tracer d'abord un contour noir épais puis le texte en couleur par dessus.
                cv2.putText(
                    frame,
                    pred_text,
                    (smooth_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8, # Taille de lettre
                    (0, 0, 0),
                    5, # Epais
                    cv2.LINE_AA,
                )

                # Texte
                cv2.putText(
                    frame,
                    pred_text,
                    (smooth_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    text_color,
                    2,
                    cv2.LINE_AA,
                )

                # Emoji
                if pred_key in gesture_images: # Si la correction par pc (pred_key) est dans dictionnaire d'émoji

                    # Composer d'émoiji PNG transparents en utilisant la fonction à la ligne 142
                    overlay_emoji_transparent(
                        frame, gesture_images[pred_key], emoji_x, emoji_y, size=120
                    )

            except Exception as e:
                print("Erreur :", e)

    # Si la main détecte pas sur l'écran, if not len(hands_sorted) == 1
    else:

        is_first_frame = True

    # Afficher le frame l'écran pour on puisse visualiser
    cv2.imshow("TIAGO Robot Recognition", frame)

    # Touche ESC pour s'arrêter en utilisant le code ASCII
    if cv2.waitKey(1) & 0xFF == 27:
        break

# FIN
cap.release() # On ferme le webcam
cv2.destroyAllWindows() # Fermer de force toutes les fenêtres liées à OpenCV (« Collecte Multi-Mains ») qui étaient ouvertes à l'écran et effacer les complètement de l'espace mémoire de l'ordinateur.
