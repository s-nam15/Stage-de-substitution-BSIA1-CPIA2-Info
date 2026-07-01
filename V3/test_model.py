import cv2 # OpenCV -> d'accèder à la webcam et de dessiner du texte
import mediapipe as mp # détecter les mains et extraire les 21 landmarks
import joblib # Enregistrer le modèle machine learning entrâiné sous forme de fichier (.pkl)
import os # Permettre de gérer les chemins de fichiers
import numpy as np # Pour calculer l'algèbre linéaire
import math # Pour calculer la distance euclidienne
from pyniryo import NiryoRobot ### AJOUT NIRYO : Importation de l'API officielle

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl") 
IMG_DIR = os.path.join(BASE_DIR, "static", "img") 

# Chargement du modèle
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé.")
else:
    print(f"❌ Modèle introuvable à : {MODEL_PATH}")
    exit()

### AJOUT NIRYO : Connexion au robot via l'IP Wi-Fi validée + Calibrage Automatique
print("🔄 Connexion au Niryo Ned (10.10.10.10)...")
try:
    robot = NiryoRobot("10.10.10.10")
    print("✅ Robot connecté avec succès !")
    
    print("⚙️ Calibrage du robot en cours (attends qu'il bouge un peu)...")
    robot.calibrate_auto()
    print("✅ Calibrage terminé, prêt à recevoir les gestes !")

    # On mémorise la position exacte de fin de calibration
    position_apres_cali = robot.get_joints()
    print(f"📍 Position de calibration enregistrée : {position_apres_cali}")

except Exception as e:
    print(f"❌ Impossible de se connecter au robot : {e}")
    exit()

# Stockage du dernier geste pour éviter d'envoyer en boucle le même ordre (uniquement pour les gestes fixes)
dernier_geste_execute = None

# Chargement des images depuis dans un dossier img
gesture_images = {} 
for ml_label, file_name in mapping.items():
    path = os.path.join(IMG_DIR, f"{file_name}.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        gesture_images[ml_label] = img

def normalize_hand(hand_landmarks):
    features = []
    wrist = hand_landmarks.landmark[0] 
    middle = hand_landmarks.landmark[12] 

    hand_size = math.sqrt(
        (middle.x - wrist.x) ** 2
        + (middle.y - wrist.y) ** 2
        + (middle.z - wrist.z) ** 2
    )

    if hand_size == 0:
        hand_size = 1

    for lm in hand_landmarks.landmark:
        x = (lm.x - wrist.x) / hand_size 
        y = (lm.y - wrist.y) / hand_size 
        z = (lm.z - wrist.z) / hand_size 
        features.extend([x, y, z]) 

    return features 

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

smooth_x, smooth_y = 0, 0
is_first_frame = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h_f, w_f, _ = frame.shape 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb) 

    if result.multi_hand_landmarks:
        test_features = [] 
        hands_sorted = sorted(
            result.multi_hand_landmarks, key=lambda h: h.landmark[0].x 
        )

        for hl in hands_sorted:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS) 
            normalized = normalize_hand(hl)
            test_features.extend(normalized)

        if len(hands_sorted) == 1:
            test_features.extend([0.0] * 63) 

        if len(test_features) == 126: 
            try:
                probabilities = model.predict_proba([test_features])[0] 
                max_prob = np.max(probabilities)
                percentage = int(max_prob * 100)

                # Geste connu (Supérieur ou égal à 70%)
                if max_prob >= CONFIDENCE_THRESHOLD: 
                    class_idx = np.argmax(probabilities) 
                    pred_key = model.classes_[class_idx]

                    if pred_key in mapping:
                        clean_name = mapping[pred_key].replace("_", " ").capitalize()
                        pred_text = f"{clean_name} ({percentage}%)"
                    else:
                        pred_text = f"{pred_key.replace('_', ' ').capitalize()} ({percentage}%)"

                    text_color = (0, 255, 0) 

                    # Récupération immédiate des angles actuels pour le suivi temps réel
                    try:
                        angles_actuels = robot.get_joints()
                    except Exception:
                        angles_actuels = [0.0] * 6

                    # 1. GESTES UNIQUES (Exécutés une seule fois au changement de geste)
                    if pred_key != dernier_geste_execute:
                        print(f"🤖 Action détectée : {pred_key}")
                        
                        if pred_key == "PALMS_TOGETHER": # Paume contre paume doigts vers le haut
                            # Retour à l'origine verticale
                            robot.move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                            dernier_geste_execute = pred_key

                        elif pred_key == "RAISED_HANDS": # Mains levées
                            # Retour à la position enregistrée après calibration
                            robot.move_joints(position_apres_cali)
                            dernier_geste_execute = pred_key

                            # Au cas où quand on deconnecte le robot avec la position non calibration
                            #robot.move_joints([0.01011087290810508, 0.4934822912792142, -1.2980141354100267, 9.265358979293481e-05, -0.16116063631778532, -0.012232660594410305])

                        elif pred_key == "OPEN_HANDS" or pred_key == "SPREAD_HAND": # Main levée ou Main levée doigts écartés
                            # Ouvrir la pince
                            robot.open_gripper()
                            dernier_geste_execute = pred_key
                        
                        # Poing levée ou Poing de face ou Poing à droite ou Poing à gauche
                        elif pred_key == "RAISED_FIST" or pred_key == "FRONT_FIST" or pred_key == "FIST_RIGHT" or pred_key == "FIST_LEFT": 
                            # Fermer la pince
                            robot.close_gripper()
                            dernier_geste_execute = pred_key

                    # 2. GESTES CONTINUS (Exécutés en boucle pour le suivi en temps réel)
                    if pred_key == "POINT_LEFT": # Main avec index pointant à gauche
                        # Ajoute un petit pas à la base (Axe 1) pour tourner à gauche
                        angles_actuels[0] += 0.16 
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None # Réinitialise pour autoriser la répétition immédiate

                    elif pred_key == "POINT_RIGHT": # Main avec index pointant à droite
                        # Retire un petit pas à la base (Axe 1) pour tourner à droite
                        angles_actuels[0] -= 0.16 
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "POINT_UP": # Main avec index pointant vers le haut
                        # Redresse l'épaule (Axe 2) pour reculer et lever le bras
                        angles_actuels[1] += 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None
                    
                    elif pred_key == "POINT_AT_USER": # Main avec index pointant vers le bas
                        # Penches l'épaule (Axe 2) pour avancer et descendre le bras
                        angles_actuels[1] -= 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "THUMBS_UP": # Pouce vers le haut
                        # Axe 3 (Coude) -> Plie le coude vers le haut
                        angles_actuels[2] += 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "THUMBS_DOWN": # Pouce vers le bas
                        # Axe 3 (Coude) -> Déplie le coude vers le bas
                        angles_actuels[2] -= 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "HAND_TO_THE_LEFT": # Main vers la gauche
                        # Axe 4 (Poignet) -> Rotation dans le sens anti-horaire
                        angles_actuels[3] -= 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "HAND_TO_THE_RIGHT": # Main vers la droite
                        # Axe 4 (Poignet) -> Rotation dans le sens horaire
                        angles_actuels[3] += 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "PALM_UP": # Main paume vers le haut
                        # Axe 5 (Poignet) -> Oriente la pince vers le haut
                        angles_actuels[4] += 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "PALM_DOWN": # Main paume vers le bas
                        # Axe 5 (Poignet) -> Oriente la pince vers le bas
                        angles_actuels[4] -= 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "PINCHED_FINGERS": # Pouce et index rapprochés
                        # Axe 6 (Pince) -> Rotation horaire de la pince
                        angles_actuels[5] += 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                    elif pred_key == "FINGERS_JOINED": # Bout des doigts joints
                        # Axe 6 (Pince) -> Rotation anti-horaire de la pince
                        angles_actuels[5] -= 0.12
                        robot.move_joints(angles_actuels)
                        dernier_geste_execute = None

                # Geste inconnu (Inférieur à 70%)
                else:
                    pred_key = "INCONNU"
                    class_idx = np.argmax(probabilities)
                    potential_key = model.classes_[class_idx]
                    
                    if potential_key in mapping:
                        clean_name = mapping[potential_key].replace("_", " ").capitalize()
                    else:
                        clean_name = potential_key
                        
                    pred_text = f"Inconnu... ({clean_name} ? {percentage}%)"
                    text_color = (0, 0, 255) 

                target_x = int(hands_sorted[0].landmark[8].x * w_f)
                target_y = int(hands_sorted[0].landmark[8].y * h_f)

                if is_first_frame:
                    smooth_x = target_x
                    smooth_y = target_y
                    is_first_frame = False
                else:
                    smooth_x = int(smooth_x + 0.25 * (target_x - smooth_x))
                    smooth_y = int(smooth_y + 0.25 * (target_y - smooth_y))

                text_y = smooth_y - 40
                emoji_y = smooth_y - 180
                emoji_x = smooth_x - 60

                cv2.putText(
                    frame, pred_text, (smooth_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA,
                )

                cv2.putText(
                    frame, pred_text, (smooth_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color, 2, cv2.LINE_AA,
                )

                if pred_key in gesture_images: 
                    overlay_emoji_transparent(
                        frame, gesture_images[pred_key], emoji_x, emoji_y, size=120
                    )

            except Exception as e:
                print("Erreur :", e)

    else:
        is_first_frame = True

    cv2.imshow("TIAGO Robot Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Fermeture proprement
cap.release() 
cv2.destroyAllWindows()
robot.close_connection()