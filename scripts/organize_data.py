import os
import shutil
import random

SRC = os.path.join(os.path.dirname(__file__), "..", "data", "archive (2)", "asl_alphabet_train", "asl_alphabet_train")
TRAIN_DST = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train")
VAL_DST = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "val")
VAL_SPLIT = 0.2

random.seed(42)

classes = [d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))]
print(f"Found {len(classes)} classes: {sorted(classes)}")

for cls in classes:
    cls_src = os.path.join(SRC, cls)
    images = [f for f in os.listdir(cls_src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    split = int(len(images) * VAL_SPLIT)
    val_imgs = images[:split]
    train_imgs = images[split:]

    for dst_dir, imgs in [(TRAIN_DST, train_imgs), (VAL_DST, val_imgs)]:
        cls_dst = os.path.join(dst_dir, cls)
        os.makedirs(cls_dst, exist_ok=True)
        for img in imgs:
            shutil.copy2(os.path.join(cls_src, img), os.path.join(cls_dst, img))

    print(f"  {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

print("\nDone! Data organized into data/processed/train and data/processed/val")
