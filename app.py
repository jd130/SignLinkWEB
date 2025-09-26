import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# --------- MODEL + LABELS ----------
MODEL_PATH = "my_model_hand_only.keras"
LABELS_PATH = "labels_hand_only.npy"

model = load_model(MODEL_PATH)
actions = np.load(LABELS_PATH)  # must match training labels

app = Flask(__name__)

# --------- MEDIAPIPE HANDS (to match training: hand landmarks only) ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Buffers
sequence = deque(maxlen=30)      # for sequence of frames
sentence = []                    # final sentence string
last_pred = None
prediction_window = deque(maxlen=7)  # majority voting window

# Confidence threshold
CONF_THRESHOLD = 0.90


def extract_keypoints(results):
    """Extract LH + RH → 126 features. Missing ones = zeros."""
    lh = np.zeros(21 * 3)   # 63 features
    rh = np.zeros(21 * 3)   # 63 features

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if i == 0:
                rh = hand_array
            elif i == 1:
                lh = hand_array

    return np.concatenate([lh, rh])  # (126,)


# ---------- UI ROUTES ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/sign")
def sign_page():
    return render_template("sign.html")

@app.route("/voice")
def voice_page():
    return render_template("voice.html")


# ---------- SIGN -> TEXT (frame prediction) ----------
@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    global last_pred, sentence, prediction_window

    if 'frame' not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    # Read frame
    file = request.files['frame']
    arr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Run Mediapipe Hands for landmark detection
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    if len(sequence) == sequence.maxlen:
        seq_np = np.expand_dims(sequence, axis=0)  # (1,30,126)
        preds = model.predict(seq_np, verbose=0)[0]

        pred_class = str(actions[np.argmax(preds)])
        conf = float(np.max(preds))

        # Debug log
        print("Predicted:", pred_class, "Conf:", conf)

        # Append to prediction window
        prediction_window.append(pred_class)

        # Stability rule: high conf + majority voting
        if conf >= CONF_THRESHOLD and prediction_window.count(pred_class) >= 3:
            if pred_class.lower() == "space":
                sentence.append(" ")
                last_pred = "space"
            elif pred_class != last_pred:  # avoid repetition
                sentence.append(pred_class)
                last_pred = pred_class

        # Limit final sentence length
        if len(sentence) > 50:
            sentence = sentence[-50:]

        return jsonify({
            "prediction": pred_class,
            "sentence": "".join(sentence)
        })

    return jsonify({
        "prediction": None,
        "sentence": "".join(sentence)
    })


# ---------- RESET SENTENCE ----------
@app.route("/reset_sentence", methods=["POST"])
def reset_sentence():
    global sentence, last_pred, prediction_window
    sentence = []
    last_pred = None
    prediction_window.clear()
    return jsonify({"status": "reset"})


# ---------- VOICE -> SIGN ----------
@app.route("/voice_to_sign", methods=["POST"])
def voice_to_sign():
    """Convert voice/text input into sign image URLs"""
    data = request.get_json()
    sentence_text = data.get("text", "")

    signs = []
    for ch in sentence_text.upper():
        if ch == " ":
            signs.append("/static/sign_images/SPACE.jpg")
        elif ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            signs.append(f"/static/sign_images/{ch.lower()}.jpg")
        else:
            signs.append("/static/sign_images/DEL.jpg")

    return jsonify({"input": sentence_text, "signs": signs})


if __name__ == "__main__":
    app.run(debug=True)