# Paste this into a Kaggle Notebook (Python, GPU T4 x2 accelerator)
# Dataset: akash8992/asl-alphabet (add it via the Data panel on the right)
# After training, download runs/classify/weights/best.pt

from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

model.train(
    data="/kaggle/input/asl-alphabet/asl_alphabet_train",
    epochs=50,
    imgsz=224,
    batch=64,
    amp=True,         # safe on GPU
    dropout=0.2,
    lr0=0.001,
    cache="ram",
    project="/kaggle/working/runs",
    name="classify",
)

print("Done! Download runs/classify/weights/best.pt from the Output panel.")
