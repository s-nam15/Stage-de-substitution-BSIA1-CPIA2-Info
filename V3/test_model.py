import cv2
import mediapipe as mp
import joblib
import os
import numpy as np
import math

# ===== CONFIGURATION =====
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

CONFIDENCE_THRESHOLD = 0.7

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")
IMG_DIR = os.path.join(BASE_DIR, "static", "img")

# ===== CHARGEMENT MODÈLE =====
if os.path.exists(MODEL_PATH):

    model = joblib.load(MODEL_PATH)

    print("✅ Modèle chargé.")

else:

    print(f"❌ Modèle introuvable à : {MODEL_PATH}")
    exit()

# ===== CHARGEMENT IMAGES =====
gesture_images = {}

for ml_label, file_name in mapping.items():

    path = os.path.join(IMG_DIR, f"{file_name}.png")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is not None:
        gesture_images[ml_label] = img


# ===== NORMALISATION =====
def normalize_hand(hand_landmarks):

    features = []

    wrist = hand_landmarks.landmark[0]
    middle = hand_landmarks.landmark[12]

    # Taille de référence
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


# ===== OVERLAY PNG TRANSPARENT =====
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


# ===== VARIABLES LISSAGE =====
smooth_x, smooth_y = 0, 0
is_first_frame = True

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# ===== WEBCAM =====
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

        # ===== TRI GAUCHE -> DROITE =====
        hands_sorted = sorted(
            result.multi_hand_landmarks, key=lambda h: h.landmark[0].x
        )

        # ===== EXTRACTION NORMALISÉE =====
        for hl in hands_sorted:

            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            normalized = normalize_hand(hl)

            test_features.extend(normalized)

        # ===== UNE MAIN =====
        if len(hands_sorted) == 1:
            test_features.extend([0.0] * 63)

        # ===== PRÉDICTION =====
        if len(test_features) == 126:

            try:

                probabilities = model.predict_proba([test_features])[0]

                max_prob = np.max(probabilities)

                # ===== GESTE CONNU =====
                if max_prob >= CONFIDENCE_THRESHOLD:

                    class_idx = np.argmax(probabilities)

                    pred_key = model.classes_[class_idx]

                    if pred_key in mapping:

                        pred_text = mapping[pred_key].replace("_", " ").capitalize()

                    else:

                        pred_text = pred_key.replace("_", " ").capitalize()

                    text_color = (0, 255, 0)

                # ===== GESTE INCONNU =====
                else:

                    pred_key = "INCONNU"

                    pred_text = "Geste inconnu"

                    text_color = (0, 0, 255)

                # ===== POSITION =====
                target_x = int(hands_sorted[0].landmark[8].x * w_f)

                target_y = int(hands_sorted[0].landmark[8].y * h_f)

                # ===== LISSAGE =====
                if is_first_frame:

                    smooth_x = target_x
                    smooth_y = target_y

                    is_first_frame = False

                else:

                    smooth_x = int(smooth_x + 0.25 * (target_x - smooth_x))

                    smooth_y = int(smooth_y + 0.25 * (target_y - smooth_y))

                # ===== POSITIONS UI =====
                text_y = smooth_y - 40

                emoji_y = smooth_y - 180

                emoji_x = smooth_x - 60

                # ===== CONTOUR TEXTE =====
                cv2.putText(
                    frame,
                    pred_text,
                    (smooth_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )

                # ===== TEXTE =====
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

                # ===== EMOJI =====
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

cap.release()
cv2.destroyAllWindows()
