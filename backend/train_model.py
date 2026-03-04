"""
Train YOLOv8 on your cloth defect dataset.

BEFORE RUNNING:
  pip install ultralytics albumentations

DATASET STRUCTURE EXPECTED:
  datasets/
    cloth_defects/
      images/
        train/   ← your training images (.jpg/.png)
        val/     ← your validation images
      labels/
        train/   ← YOLO format .txt files (one per image)
        val/
      data.yaml  ← dataset config (auto-generated below)

YOLO LABEL FORMAT (per line in .txt):
  class_id cx cy width height
  (all values normalized 0–1)

CLASSES:
  0 = hole
  1 = stain
  2 = tear
  3 = fray
"""

import os
import yaml
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_DIR  = "./datasets/cloth_defects"
OUTPUT_DIR   = "./runs/cloth_defect_train"
MODEL_BASE   = "yolov8n.pt"   # Start small: yolov8n, yolov8s, yolov8m
EPOCHS       = 100
IMAGE_SIZE   = 640
BATCH_SIZE   = 16
DEVICE       = "0"  # GPU id, or "cpu"

CLASSES = ["hole", "stain", "tear", "fray"]


def create_data_yaml():
    """Auto-generate the dataset YAML for YOLO training."""
    data = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASSES),
        "names": CLASSES
    }
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"✅ data.yaml written to {yaml_path}")
    return yaml_path


def train():
    from ultralytics import YOLO

    yaml_path = create_data_yaml()

    print(f"\n🚀 Starting YOLOv8 training...")
    print(f"   Model   : {MODEL_BASE}")
    print(f"   Epochs  : {EPOCHS}")
    print(f"   Img size: {IMAGE_SIZE}")
    print(f"   Dataset : {DATASET_DIR}\n")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data      = yaml_path,
        epochs    = EPOCHS,
        imgsz     = IMAGE_SIZE,
        batch     = BATCH_SIZE,
        device    = DEVICE,
        project   = OUTPUT_DIR,
        name      = "exp",
        patience  = 20,           # Early stopping
        augment   = True,         # Built-in augmentation
        mosaic    = 1.0,          # Mosaic augmentation
        degrees   = 10.0,         # Rotation
        fliplr    = 0.5,          # Horizontal flip
        flipud    = 0.2,          # Vertical flip
        hsv_h     = 0.02,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        save      = True,
        plots     = True,
    )

    print("\n✅ Training complete!")
    best_model = os.path.join(OUTPUT_DIR, "exp/weights/best.pt")
    if os.path.exists(best_model):
        import shutil
        dest = "../../models/cloth_defect.pt"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(best_model, dest)
        print(f"✅ Best model copied to: {dest}")
    else:
        print(f"⚠️  Model saved at: {OUTPUT_DIR}/exp/weights/best.pt")

    # Evaluate
    print("\n📊 Running validation...")
    metrics = model.val()
    print(f"   mAP50     : {metrics.box.map50:.3f}")
    print(f"   mAP50-95  : {metrics.box.map:.3f}")
    print(f"   Precision : {metrics.box.mp:.3f}")
    print(f"   Recall    : {metrics.box.mr:.3f}")


def test_inference(image_path: str):
    """Quick test — run inference on a single image."""
    from ultralytics import YOLO
    model = YOLO("../../models/cloth_defect.pt")
    results = model(image_path, conf=0.35)
    for r in results:
        r.show()  # Display annotated image
        print(f"Detections: {len(r.boxes)}")
        for box in r.boxes:
            print(f"  Class {int(box.cls[0])}, Conf {float(box.conf[0]):.2f}, Box {box.xyxy[0].tolist()}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) < 3:
            print("Usage: python train_model.py test path/to/image.jpg")
        else:
            test_inference(sys.argv[2])
    else:
        train()