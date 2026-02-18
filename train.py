"""
train.py — Train YOLOv8 on multi-state Indian driving licenses

Detects 3 fields: name, dl_no, dob
Issue date and expiry date are extracted via OCR parsing (not YOLO detection).

Usage:
    python train.py              # start training
    python train.py --resume     # resume interrupted training
    python train.py --test IMG   # test trained model on one image
"""

import argparse
import os
from pathlib import Path
import torch
import cv2

DATA_YAML = "data.yaml"
WEIGHTS_OUT = "runs/detect/runs/detect/multistate_dl2/weights/best.pt"

CLASS_NAMES = {
    0: "name",
    1: "dl_no",
    2: "dob",
}


def get_device():
    """Auto-detect GPU or fallback to CPU."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        print(f"  🖥️  GPU: {name} ({vram} MB VRAM)")
        return 0
    print("  🖥️  No CUDA GPU — using CPU (much slower)")
    return "cpu"


def train(resume=False):
    """Train YOLO detector on multi-state license images."""
    from ultralytics import YOLO

    print("=" * 60)
    print("Training Multi-State License Field Detector (YOLO)")
    print("  Fields: name, dl_no, dob")
    print("  Issue/expiry dates extracted via OCR (not detected by YOLO)")
    print("=" * 60)

    device = get_device()

    # Use yolov8s — good balance of speed/accuracy
    weights = WEIGHTS_OUT if (resume and Path(WEIGHTS_OUT).exists()) else "yolov8s.pt"
    model = YOLO(weights)

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,              # RTX 2070 8GB safe batch size
        device=device,
        name="multistate_dl",
        project="runs/detect",
        exist_ok=resume,
        workers=0,            # avoid Windows multiprocessing crash

        # Learning rate
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,

        # Regularisation
        weight_decay=0.0005,
        dropout=0.0,
        patience=20,

        # Augmentation — conservative for text fields
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.4,
        degrees=3.0,          # small rotation
        translate=0.05,
        scale=0.3,
        shear=1.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.0,           # NEVER flip — text becomes mirrored
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.2,

        # Output
        save=True,
        save_period=10,
        val=True,
        plots=True,
        close_mosaic=15,
    )

    print(f"\n✅ Training complete!")
    print(f"   Best weights → {WEIGHTS_OUT}")
    print(f"\nTest on an image:")
    print(f"   python train.py --test path/to/license.jpg")


def test_image(image_path):
    """Visual test — draws detected boxes and saves result."""
    from ultralytics import YOLO

    if not Path(WEIGHTS_OUT).exists():
        print(f"❌ Weights not found: {WEIGHTS_OUT}")
        print("   Run training first: python train.py")
        return

    model = YOLO(WEIGHTS_OUT)
    device = get_device()

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read {image_path}")
        return

    h, w = img.shape[:2]
    results = model.predict(image_path, conf=0.25, device=device, verbose=False)
    boxes = results[0].boxes

    COLORS = {
        "name": (0, 255, 0),
        "dl_no": (255, 0, 0),
        "dob": (0, 255, 255),
    }

    print(f"\n{'='*50}")
    print(f"Detected {len(boxes)} fields in {Path(image_path).name}")
    print(f"{'='*50}")

    for box in boxes:
        cls_id = int(box.cls[0])
        field = CLASS_NAMES.get(cls_id, f"class{cls_id}")
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = COLORS.get(field, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{field} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        print(f"  {field:10s}: conf={conf:.2f}  box=[{x1},{y1},{x2},{y2}]")

    out = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(out, img)
    print(f"\n💾 Saved: {out}")
    os.startfile(str(Path(out).absolute()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Resume interrupted training")
    ap.add_argument("--test", type=str, metavar="IMAGE",
                    help="Test trained model on a single image")
    args = ap.parse_args()

    if args.test:
        test_image(args.test)
    else:
        train(resume=args.resume)
