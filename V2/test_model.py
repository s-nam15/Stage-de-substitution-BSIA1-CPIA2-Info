import cv2
import mediapipe as mp
import joblib
import os
import math
import time

# ===== CONFIGURATION =====
mapping = {
    "FINGERS_JOINED": "bout_des_doigts_joints",
    "HORNS": "cornes_avec_les_doigts",
    "MIDDLE_FINGER": "doigt_dhonneur",
    "CROSSED_FINGERS": "doigts_croises",
    "POINT_UP": "index_pointant_vers_le_haut",
    "POINT_AT_USER": "index_pointant_vers_lutilisateur",
    "LOVE_YOU": "signe_je_taime",
    "POINT_RIGHT": "main_avec_index_pointant_a_droite",
    "POINT_LEFT": "main_avec_index_pointant_a_gauche",
    "POINT_DOWN": "main_avec_index_pointant_vers_le_bas",
    "POINT_UP_HAND": "main_avec_index_pointant_vers_le_haut",
    "CROSSED_THUMB_INDEX": "main_avec_index_et_pouce_croises",
    "PALM_DOWN": "main_paume_vers_le_bas",
    "PALM_UP": "main_paume_vers_le_haut",
    "RAISED_HAND": "main_levee",
    "SPREAD_HAND": "main_levee_doigts_ecartes",
    "PRAY_HANDS": "mains_en_priere",
    "RAISED_HANDS": "mains_levees",
    "OPEN_HANDS": "mains_ouvertes",
    "HEART_HANDS": "mains_qui_forment_un_coeur",
    "OK": "ok",
    "PALMS_TOGETHER": "paume_contre_paume_doigts_vers_le_haut",
    "FIST_RIGHT": "poing_a_droite",
    "FIST_LEFT": "poing_a_gauche",
    "FRONT_FIST": "poing_de_face",
    "RAISED_FIST": "poing_leve",
    "HANDSHAKE": "poignee_de_main",
    "PINCHED_FINGERS": "pouce_et_index_rapproches",
    "THUMBS_DOWN": "pouce_vers_le_bas",
    "THUMBS_UP": "pouce_vers_le_haut",
    "VULCAN": "salut_vulcain",
    "CALL_ME": "signe_appel_telephonique_avec_les_doigts",
    "PEACE": "v_de_la_victoire",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")
IMG_DIR = os.path.join(BASE_DIR, "img")

# ===== CHARGEMENT MODÈLE =====
model = joblib.load(MODEL_PATH)

# ===== CHARGEMENT IMAGES =====
gesture_images = {}

for ml_label, file_name in mapping.items():

    path = os.path.join(IMG_DIR, f"{file_name}.png")

    img = cv2.imread(path)

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

    if hand_size == 0:
        hand_size = 1

    for lm in hand_landmarks.landmark:

        x = (lm.x - wrist.x) / hand_size
        y = (lm.y - wrist.y) / hand_size
        z = (lm.z - wrist.z) / hand_size

        features.extend([x, y, z])

    return features


# ===== OVERLAY =====
def overlay_emoji(frame, img, x, y, size=120):

    if img is None:
        return

    try:

        img_res = cv2.resize(img, (size, size))

        h, w, _ = img_res.shape

        y1 = max(0, y)
        y2 = min(frame.shape[0], y + h)

        x1 = max(0, x)
        x2 = min(frame.shape[1], x + w)

        frame[y1:y2, x1:x2] = img_res[0 : y2 - y1, 0 : x2 - x1]

    except:
        pass


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

        # ===== UNE SEULE MAIN =====
        if len(hands_sorted) == 1:
            test_features.extend([0.0] * 63)

        # ===== PRÉDICTION =====
        if len(test_features) == 126:

            try:

                pred = model.predict([test_features])[0]

                # ===== POSITION =====
                idx_x = int(hands_sorted[0].landmark[8].x * w_f)

                idx_y = int(hands_sorted[0].landmark[8].y * h_f)

                text_y = idx_y - 40
                emoji_y = idx_y - 200

                # ===== FOND TEXTE =====
                cv2.rectangle(
                    frame,
                    (idx_x - 10, text_y - 30),
                    (idx_x + 300, text_y + 10),
                    (0, 0, 0),
                    -1,
                )

                # ===== TEXTE =====
                cv2.putText(
                    frame,
                    pred,
                    (idx_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # ===== EFFET PULSE =====
                pulse = int(15 * math.sin(time.time() * 6))

                size = 120 + pulse

                # ===== IMAGE =====
                if pred in gesture_images:

                    overlay_emoji(
                        frame, gesture_images[pred], idx_x - 60, emoji_y, size=size
                    )

            except Exception as e:
                print("Erreur :", e)

    cv2.imshow("TIAGO Robot Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
