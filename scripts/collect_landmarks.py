"""
Collect MediaPipe hand-landmark samples for each ASL letter.

Usage:
    python scripts/collect_landmarks.py

Controls:
    SPACE  — capture current frame manually
    N      — skip to next letter
    Q      — save and quit

Auto-capture fires every 0.6s when a hand is detected — just hold each sign steady.
"""
import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import csv
import os
import sys
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "landmarks")
os.makedirs(DATA_DIR, exist_ok=True)
OUT_CSV = os.path.join(DATA_DIR, "samples.csv")

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space"]
SAMPLES_PER_CLASS = 50
AUTO_CAPTURE_INTERVAL = 0.6  # seconds between auto-captures

mp_hands = mp.solutions.hands  # type: ignore
mp_draw = mp.solutions.drawing_utils  # type: ignore
hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)


def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    pts -= pts[0]           # center on wrist
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts.flatten().tolist()


# ── Load counts from any existing CSV ──────────────────────────────────────
counts = {l: 0 for l in LETTERS}
if os.path.exists(OUT_CSV):
    with open(OUT_CSV, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0] in counts:
                counts[row[0]] += 1

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not found.")
    sys.exit(1)

# Start at the first letter that still needs samples
current_idx = 0
for i, letter in enumerate(LETTERS):
    if counts[letter] < SAMPLES_PER_CLASS:
        current_idx = i
        break
else:
    print("All classes already have enough samples. Run train_landmark.py.")
    cap.release()
    sys.exit(0)

print(f"Collecting {SAMPLES_PER_CLASS} samples per class ({len(LETTERS)} classes)")
print("Auto-capture is ON — hold each sign steady.  SPACE=manual  N=skip  Q=quit")

last_capture_time = 0.0
flash_until = 0.0        # show green flash after capture

with open(OUT_CSV, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if os.path.getsize(OUT_CSV) == 0:
        header = ["label"] + [f"{ax}{i}" for i in range(21) for ax in "xyz"]
        writer.writerow(header)

    while cap.isOpened() and current_idx < len(LETTERS):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = hand_detector.process(rgb)

        letter = LETTERS[current_idx]
        count = counts[letter]
        now = time.time()

        hand_landmarks = None
        if mp_result.multi_hand_landmarks:
            hand_landmarks = mp_result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # type: ignore

        # ── Auto-capture ───────────────────────────────────────────────────
        captured_this_frame = False
        if hand_landmarks and (now - last_capture_time) >= AUTO_CAPTURE_INTERVAL:
            vec = normalize_landmarks(hand_landmarks)
            writer.writerow([letter] + vec)
            csvfile.flush()
            counts[letter] += 1
            count = counts[letter]
            last_capture_time = now
            flash_until = now + 0.15
            captured_this_frame = True

            if count >= SAMPLES_PER_CLASS:
                print(f"  {letter}: {count} samples — done")
                current_idx += 1
                last_capture_time = now  # reset timer for next letter
                time.sleep(0.4)
                continue

        # ── HUD ───────────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Dark top banner
        banner = frame.copy()
        cv2.rectangle(banner, (0, 0), (w, 105), (0, 0, 0), -1)
        frame = cv2.addWeighted(banner, 0.55, frame, 0.45, 0)

        # Letter name
        cv2.putText(frame, f"Sign:  {letter}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (0, 255, 0) if now < flash_until else (255, 255, 255), 3)

        # Progress bar
        bar_w = w - 40
        fill = int(bar_w * min(count, SAMPLES_PER_CLASS) / SAMPLES_PER_CLASS)
        cv2.rectangle(frame, (20, 78), (20 + bar_w, 98), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 78), (20 + fill, 98), (0, 210, 80), -1)
        cv2.putText(frame, f"{count}/{SAMPLES_PER_CLASS}", (w - 100, 97),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # Overall progress
        total_done = sum(min(v, SAMPLES_PER_CLASS) for v in counts.values())
        total_needed = SAMPLES_PER_CLASS * len(LETTERS)
        cv2.putText(frame, f"Overall: {total_done}/{total_needed}", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Hand status
        if hand_landmarks:
            cv2.putText(frame, "Hand: detected", (20, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)
        else:
            cv2.putText(frame, "No hand — show your hand", (20, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        cv2.putText(frame, "SPACE=capture  N=skip  Q=quit", (w - 280, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        cv2.imshow("Landmark Collection — Sign2Speech", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            print(f"  {letter}: skipped ({count} samples so far)")
            current_idx += 1
            last_capture_time = now
        elif key == ord(" ") and hand_landmarks:
            if not captured_this_frame:
                vec = normalize_landmarks(hand_landmarks)
                writer.writerow([letter] + vec)
                csvfile.flush()
                counts[letter] += 1
                count = counts[letter]
                last_capture_time = now
                flash_until = now + 0.2
                if count >= SAMPLES_PER_CLASS:
                    print(f"  {letter}: {count} samples — done")
                    current_idx += 1

cap.release()
hand_detector.close()
cv2.destroyAllWindows()

collected = {k: v for k, v in counts.items() if v > 0}
print(f"\nSaved to: {OUT_CSV}")
print(f"Samples per class: {collected}")
print("\nNext: python scripts/train_landmark.py")
