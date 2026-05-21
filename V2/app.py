from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import joblib
import os
import numpy as np
import math
import time

app = Flask(__name__)

# =======================
# CONFIG PATHS
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")
IMG_DIR = os.path.join(BASE_DIR, "img")

# =======================
# MODEL
# =======================
model = joblib.load(MODEL_PATH)

# =======================
# MAPPING IMAGES
# =======================
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
    "PEACE": "v_de_la_victoire",
}

gesture_images = {}

for label, file in mapping.items():
    path = os.path.join(IMG_DIR, file + ".png")
    img = cv2.imread(path)

    # debug utile
    if img is None:
        print("Image manquante:", path)
    else:
        gesture_images[label] = img

# =======================
# MEDIAPIPE
# =======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# =======================
# OVERLAY IMAGE
# =======================
def overlay(frame, img, x, y, size=120):
    if img is None:
        return

    img = cv2.resize(img, (size, size))
    h, w, _ = img.shape

    y1, y2 = max(0, y), min(frame.shape[0], y + h)
    x1, x2 = max(0, x), min(frame.shape[1], x + w)

    frame[y1:y2, x1:x2] = img[0 : y2 - y1, 0 : x2 - x1]


# =======================
# FEATURES EXTRACTION
# =======================
def extract_features(result):
    features = []

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for lm in hand.landmark:
                features.extend([lm.x, lm.y, lm.z])

    # pad si une seule main
    if len(features) == 63:
        features.extend([0.0] * 63)

    return features if len(features) == 126 else None


# =======================
# GENERATE STREAM
# =======================
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        pred = None

        # -------- PREDICTION --------
        features = extract_features(result)

        if features:
            try:
                pred = model.predict([features])[0]
            except:
                pred = None

        # -------- LANDMARKS --------
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # -------- OVERLAY --------
        if pred and result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            idx_x = int(hand.landmark[8].x * w)
            idx_y = int(hand.landmark[8].y * h)

            # texte
            cv2.putText(
                frame,
                pred,
                (idx_x, idx_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # animation
            pulse = int(10 * math.sin(time.time() * 8))
            size = 120 + pulse

            if pred in gesture_images:
                overlay(frame, gesture_images[pred], idx_x - 60, idx_y - 200, size=size)

        # -------- ENCODE STREAM --------
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# =======================
# ROUTES FLASK
# =======================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =======================
# RUN
# =======================
if __name__ == "__main__":
    app.run(debug=False)
