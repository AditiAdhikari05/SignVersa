"""
Train a RandomForest classifier on collected MediaPipe landmark samples.

Usage:
    python scripts/train_landmark.py

Reads:  data/landmarks/samples.csv
Writes: models/landmark_clf.pkl
"""
import csv
import os
import sys
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "landmarks", "samples.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "landmark_clf.pkl")

if not os.path.exists(DATA_PATH):
    print(f"No data at {DATA_PATH}\nRun collect_landmarks.py first.")
    sys.exit(1)

# ── Load ───────────────────────────────────────────────────────────────────
X, y = [], []
with open(DATA_PATH, newline="") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) < 2:
            continue
        y.append(row[0])
        X.append([float(v) for v in row[1:]])

X = np.array(X, dtype=np.float32)
y = np.array(y)

classes, counts = np.unique(y, return_counts=True)
print(f"Dataset: {len(X)} samples, {len(classes)} classes")
for cls, cnt in zip(classes, counts):
    print(f"  {cls:8s}: {cnt} samples")

if len(classes) < 2:
    print("Need at least 2 classes. Collect more data.")
    sys.exit(1)

# ── Train ──────────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    import pickle
except ImportError:
    print("scikit-learn not found. Run: pip install scikit-learn")
    sys.exit(1)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"\nTraining RandomForest on {len(X_train)} samples...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
val_acc = clf.score(X_val, y_val)
print(f"Validation accuracy: {val_acc:.1%}")

# Per-class accuracy
y_pred = clf.predict(X_val)
print("\nPer-class accuracy on validation set:")
for i, cls in enumerate(le.classes_):
    mask = y_val == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).mean()
        bar = "█" * int(acc * 20)
        print(f"  {cls:8s}: {acc:.0%}  {bar}")

# ── Save ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    import pickle
    pickle.dump({"clf": clf, "label_encoder": le}, f)

print(f"\nModel saved to: {MODEL_PATH}")
print("Next: python scripts/sign_to_speech.py")
