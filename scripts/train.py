import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "ultralytics"))

from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

model.train(
    data=os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
    epochs=10,
    imgsz=224,
    batch=32,
    fraction=0.3,
    project=os.path.join(os.path.dirname(__file__), "..", "runs"),
    name="classify",
)

print("Training done. Run: copy runs\\classify\\weights\\best.pt models\\base\\best.pt")
