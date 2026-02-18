"""
remap_labels.py — Remap YOLO labels from 9-class to 3-class format

Your original Roboflow dataset has 9 classes, but we only need 3:
  Original class 4 (name)   → New class 0
  Original class 2 (dl_no)  → New class 1
  Original class 3 (dob)    → New class 2

All other classes (add, blood_group, relation_with, rto, state, vehicle_type)
are discarded.

Usage:
    python remap_labels.py
"""

from pathlib import Path

# Class mapping: old_class_id → new_class_id
# Only keep name, dl_no, dob
CLASS_MAP = {
    4: 0,  # name
    2: 1,  # dl_no
    3: 2,  # dob
}

DATASET_ROOT = Path("D:/WORK/License verification easyOCR")
LABEL_FOLDERS = [
    DATASET_ROOT / "train" / "labels",
    DATASET_ROOT / "valid" / "labels",
    DATASET_ROOT / "test" / "labels",
]


def remap_label_file(label_path):
    """Remap a single label file — keep only name/dl_no/dob, discard others."""
    lines = label_path.read_text().strip().splitlines()
    new_lines = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        old_class = int(parts[0])
        if old_class not in CLASS_MAP:
            # Skip unwanted classes
            continue

        new_class = CLASS_MAP[old_class]
        coords = " ".join(parts[1:])
        new_lines.append(f"{new_class} {coords}")

    # Write back
    if new_lines:
        label_path.write_text("\n".join(new_lines) + "\n")
    else:
        # No relevant boxes — delete the label file (image will be skipped in training)
        label_path.unlink()


def main():
    total_files = 0
    remapped = 0
    deleted = 0

    for folder in LABEL_FOLDERS:
        if not folder.exists():
            print(f"⚠️  Folder not found: {folder}")
            continue

        print(f"\n📂 Processing {folder.name}/...")
        for label_file in folder.glob("*.txt"):
            total_files += 1
            original_lines = label_file.read_text().strip().splitlines()

            remap_label_file(label_file)

            if not label_file.exists():
                deleted += 1
            elif label_file.read_text().strip() != "\n".join(original_lines):
                remapped += 1

    print(f"\n{'='*60}")
    print(f"✅ Label remapping complete")
    print(f"{'='*60}")
    print(f"  Total files processed: {total_files}")
    print(f"  Files remapped:        {remapped}")
    print(f"  Files deleted (no relevant boxes): {deleted}")
    print(f"\nYou can now run:  python train.py")


if __name__ == "__main__":
    main()
