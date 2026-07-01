import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "ultralytics"))

from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

model.train(
    data=os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
    epochs=50,
    imgsz=224,
    batch=16,
    fraction=0.3,
    amp=False,        # MUST disable — AMP on CPU corrupts gradients silently
    dropout=0.2,      # prevent overfitting
    lr0=0.001,        # lower LR for stable fine-tuning
    cache="ram",      # cache images in RAM — cuts epoch time by ~40%
    project=os.path.join(os.path.dirname(__file__), "..", "runs"),
    name="classify",
)

print("Training done. Run: copy runs\\classify\\weights\\best.pt models\\base\\best.pt")
