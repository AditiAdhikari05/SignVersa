import cv2
import pyttsx3
import time
import os
import sys
import numpy as np
import mediapipe as mp  # type: ignore

# ── Classifier selection ───────────────────────────────────────────────────
# Prefer landmark model (fast, personal, no GPU needed).
# Falls back to YOLO if landmark model hasn't been trained yet.

LANDMARK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "landmark_clf.pkl")
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")

use_landmark = os.path.exists(LANDMARK_MODEL_PATH)

clf = None
label_encoder = None
yolo_model = None

if use_landmark:
    try:
        import pickle
        with open(LANDMARK_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        clf = data["clf"]
        label_encoder = data["label_encoder"]
        print(f"Landmark classifier loaded — {len(label_encoder.classes_)} classes")
    except Exception as e:
        print(f"Failed to load landmark model: {e}")
        use_landmark = False

if not use_landmark:
    if not os.path.exists(YOLO_MODEL_PATH):
        print("No model found. Run collect_landmarks.py + train_landmark.py first.")
        sys.exit(1)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "ultralytics"))
    from ultralytics import YOLO  # type: ignore
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded (landmark model not found — run collect_landmarks.py to upgrade)")

# ── MediaPipe ─────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands  # type: ignore
mp_draw = mp.solutions.drawing_utils  # type: ignore
hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ── TTS ───────────────────────────────────────────────────────────────────
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# ── Settings ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65 if use_landmark else 0.50
HOLD_SECONDS = 1.5

# ── Camera ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    sys.exit(1)

# ── State ─────────────────────────────────────────────────────────────────
word_buffer = ""
last_letter = ""
last_sign_time = time.time()
letter_registered = False

EMERGENCY = {
    "1": "I need a doctor",
    "2": "Call 911",
    "3": "I am deaf",
    "4": "I am allergic",
    "5": "Please write it down",
}


def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    pts -= pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts.flatten().reshape(1, -1)


def get_hand_crop(frame, landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]
    pad = 40
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


mode_label = "Landmark RF" if use_landmark else "YOLO"
print(f"Sign2Speech — {mode_label} + MediaPipe (fully offline)")
print("Hold a sign 1.5s to register | SPACE=speak | DEL=delete | Q=quit")
print("Emergency: 1=doctor  2=911  3=deaf  4=allergic  5=write it down")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_result = hand_detector.process(rgb)

    top1_label = ""
    confidence = 0.0
    hand_box = None

    if mp_result.multi_hand_landmarks:
        landmarks = mp_result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)  # type: ignore

        if use_landmark:
            vec = normalize_landmarks(landmarks)
            probs = clf.predict_proba(vec)[0]
            top1_idx = np.argmax(probs)
            confidence = float(probs[top1_idx])
            top1_label = label_encoder.inverse_transform([top1_idx])[0].lower()
        else:
            hand_crop, hand_box = get_hand_crop(frame, landmarks)
            if hand_crop.size > 0:
                results = yolo_model(hand_crop, verbose=False)
                top1_idx = results[0].probs.top1
                top1_label = results[0].names[top1_idx].lower()
                confidence = results[0].probs.top1conf.item()

        # Draw bounding box (landmark mode: derive from landmarks)
        if use_landmark:
            h, w = frame.shape[:2]
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
            pad = 30
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)
            hand_box = (x1, y1, x2, y2)

    if hand_box:
        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 200, 255)
        cv2.rectangle(frame, (hand_box[0], hand_box[1]), (hand_box[2], hand_box[3]), color, 2)

    now = time.time()

    if mp_result.multi_hand_landmarks is None:
        cv2.putText(frame, "Show your hand to the camera", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        last_letter = ""
        last_sign_time = now
        letter_registered = False

    elif confidence >= CONFIDENCE_THRESHOLD and top1_label not in ("nothing",):
        if top1_label == last_letter:
            held_for = now - last_sign_time
            progress = min(held_for / HOLD_SECONDS, 1.0)

            if held_for >= HOLD_SECONDS and not letter_registered:
                letter_registered = True
                if top1_label == "space":
                    if word_buffer:
                        print(f"Speaking: {word_buffer}")
                        engine.say(word_buffer)
                        engine.runAndWait()
                        word_buffer = ""
                elif top1_label == "del":
                    word_buffer = word_buffer[:-1]
                else:
                    word_buffer += top1_label.upper()
                    print(f"Added: {top1_label.upper()} → buffer: {word_buffer}")

            bar_width = int(progress * 200)
            cv2.rectangle(frame, (10, 60), (10 + bar_width, 80), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 60), (210, 80), (255, 255, 255), 2)
        else:
            last_letter = top1_label
            last_sign_time = now
            letter_registered = False

        cv2.putText(frame, f"Sign: {top1_label.upper()} ({confidence:.0%})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame,
                    f"Hand detected — {top1_label.upper()} ({confidence:.0%})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        last_letter = ""
        last_sign_time = now
        letter_registered = False

    cv2.putText(frame, f"Word: {word_buffer}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"[{mode_label}]  1=doctor 2=911 3=deaf 4=allergic 5=write | Q=quit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.imshow("Sign2Speech", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    ch = chr(key) if key < 128 else ""
    if ch in EMERGENCY:
        phrase = EMERGENCY[ch]
        print(f"Emergency: {phrase}")
        engine.say(phrase)
        engine.runAndWait()

cap.release()
hand_detector.close()
cv2.destroyAllWindows()
