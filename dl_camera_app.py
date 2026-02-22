"""
dl_camera_app.py — Multi-State Driving License Scanner (Simplified + Enhanced)

Improvements:
  1. Aggressive preprocessing (2x upscale, stronger enhancement)
  2. PaddleOCR only (removed multi-OCR to avoid crashes)
  3. Smart validation and error correction

Workflow:
  FRONT (3 retries): name, dl_no, issue_date
  BACK (3 retries): expiry_date
"""

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import re
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pymongo import MongoClient

# =========================
# Config
# =========================
YOLO_WEIGHTS = "runs/detect/runs/detect/multistate_dl2/weights/best.pt"
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://corazortechnology:A0Qfk2PbjOMKN32Z@cluster0.drxzj5r.mongodb.net/")

client = MongoClient(MONGO_URI)
collection = client["licenseDB"]["licenses"]

yolo_model = None
ocr = PaddleOCR(use_angle_cls=True, lang="en")
MAX_RETRIES = 3

YOLO_CLASSES = {0: "name", 1: "dl_no", 2: "dob"}

# =========================
# Enhanced Preprocessing
# =========================
def enhance_crop(crop):
    """Aggressive enhancement for small text."""
    h, w = crop.shape[:2]

    # 2x upscaling
    if w < 600:
        scale = 600 / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Strong denoising
    crop = cv2.fastNlMeansDenoisingColored(crop, None, 15, 15, 7, 21)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Aggressive CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)

    # Strong unsharp masking
    blur = cv2.GaussianBlur(eq, (0, 0), 2.5)
    sharp = cv2.addWeighted(eq, 2.0, blur, -1.0, 0)

    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def enhance_full_image(frame):
    """Enhanced full-image preprocessing."""
    h, w = frame.shape[:2]

    # 2x upscaling
    up = cv2.resize(frame, None, fx=2000/w, fy=2000/w, interpolation=cv2.INTER_CUBIC)
    up = cv2.fastNlMeansDenoisingColored(up, None, 15, 15, 7, 21)

    # Enhanced grayscale
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 3)
    enhanced = cv2.addWeighted(eq, 2.0, blur, -1.0, 0)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# =========================
# OCR
# =========================
def run_ocr(img):
    """Run PaddleOCR and return text."""
    try:
        result = ocr.ocr(img, cls=True)
        texts = []
        for line in result:
            if not line:
                continue
            for word_info in line:
                text = word_info[1][0].strip()
                if text:
                    texts.append(text)
        return " ".join(texts) if texts else None
    except Exception as e:
        print(f"    OCR error: {e}")
        return None


def extract_all_text(frame):
    """Run OCR on enhanced full image."""
    enhanced = enhance_full_image(frame)
    result = ocr.ocr(enhanced, cls=True)

    all_texts = []
    for line in result:
        if not line:
            continue
        for word_info in line:
            text = word_info[1][0].strip()
            if text:
                all_texts.append(text)

    return all_texts


# =========================
# Validation & Post-Processing
# =========================
def fix_ocr_errors(text):
    """Fix common OCR mistakes."""
    if not text:
        return text

    result = list(text)
    for i, ch in enumerate(result):
        prev_digit = i > 0 and result[i-1].isdigit()
        next_digit = i < len(result)-1 and result[i+1].isdigit()

        if ch in ('O', 'o') and (prev_digit or next_digit):
            result[i] = '0'
        elif ch in ('I', 'l') and (prev_digit or next_digit):
            result[i] = '1'
        elif ch == 'S' and (prev_digit or next_digit):
            result[i] = '5'

    return ''.join(result)


def validate_dl_number(text):
    """Validate DL number: XX## #### #######."""
    if not text:
        return None

    cleaned = fix_ocr_errors(text.replace(" ", "").replace("-", "").upper())

    # 2 letters + 13-17 digits
    if re.match(r'^[A-Z]{2}\d{13,17}$', cleaned):
        return cleaned

    # Try to extract if embedded
    match = re.search(r'([A-Z]{2}\d{13,17})', cleaned)
    return match.group(1) if match else None


def validate_name(text):
    """Validate name: 2+ words, alphabetic only."""
    if not text:
        return None

    cleaned = re.sub(r'[^A-Za-z\s\-]', '', text).strip()
    words = cleaned.split()

    if len(words) >= 2 and len(cleaned) >= 5:
        return cleaned.upper()

    return None


def validate_date(text):
    """Extract and validate dates."""
    if not text:
        return None

    # Try to find date patterns
    patterns = [
        r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})',
        r'(\d{8})',
        r'(\d{4})/(\d{4})',  # garbled DDMM/YYYY
    ]

    for pat in patterns:
        match = re.search(pat, text)
        if match:
            if len(match.groups()) == 2:
                date_str = match.group(1) + match.group(2)
            else:
                date_str = match.group(1)

            for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d%m%Y"]:
                try:
                    clean_str = date_str.replace("/", "").replace("-", "").replace(".", "")
                    if len(clean_str) == 8:
                        dt = datetime.strptime(clean_str, "%d%m%Y")
                    else:
                        dt = datetime.strptime(date_str.replace("-", "/").replace(".", "/"), fmt)

                    if 1950 <= dt.year <= 2060:
                        return dt.strftime("%d-%m-%Y")
                except:
                    continue

    return None


# =========================
# Date Finding
# =========================
def find_issue_date(texts):
    """Find issue date."""
    today = datetime.today()

    # Look for "issued" label
    for text in texts:
        if re.search(r"(?:issued|issue|on)", text, re.IGNORECASE):
            validated = validate_date(text)
            if validated:
                dt = datetime.strptime(validated, "%d-%m-%Y")
                if 2000 <= dt.year <= today.year:
                    return validated

    # Fallback: any date 2000-today
    for text in texts:
        validated = validate_date(text)
        if validated:
            dt = datetime.strptime(validated, "%d-%m-%Y")
            if 2000 <= dt.year <= today.year:
                return validated

    return None


def find_expiry_date(texts):
    """Find expiry date."""
    # Look for "validity" label
    for text in texts:
        if re.search(r"(?:validity|valid|expiry)", text, re.IGNORECASE):
            validated = validate_date(text)
            if validated:
                dt = datetime.strptime(validated, "%d-%m-%Y")
                if dt.year >= 2025:
                    return validated

    # Fallback: any future date
    for text in texts:
        validated = validate_date(text)
        if validated:
            dt = datetime.strptime(validated, "%d-%m-%Y")
            if dt.year >= 2025:
                return validated

    return None


def check_expiry(expiry_str):
    dt = datetime.strptime(expiry_str, "%d-%m-%Y")
    today = datetime.today()
    if dt < today:
        days = (today - dt).days
        return False, f"EXPIRED — {days} days ago ({dt.strftime('%d %b %Y')})"
    days = (dt - today).days
    return True, f"VALID — expires {dt.strftime('%d %b %Y')} ({days} days left)"


# =========================
# YOLO
# =========================
def load_yolo():
    global yolo_model
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(YOLO_WEIGHTS)
    print(f"  ✅ YOLO loaded (device={device})")


def detect_fields(img_bgr):
    """YOLO + OCR + validation."""
    h, w = img_bgr.shape[:2]
    if w < 1000:
        scale = 1000 / w
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]


    device = 0 if torch.cuda.is_available() else "cpu"
    results = yolo_model.predict(img_bgr, conf=0.25, device=device, verbose=False)
    boxes = results[0].boxes

    fields = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        field = YOLO_CLASSES.get(cls_id)
        if field == "dob" or not field:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1-8); y1 = max(0, y1-8)
        x2 = min(w, x2+8); y2 = min(h, y2+8)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        enhanced = enhance_crop(crop)
        text = run_ocr(enhanced)

        # Validate
        if field == "name":
            text = validate_name(text)
        elif field == "dl_no":
            text = validate_dl_number(text)

        if text and (field not in fields or conf > fields[field].get("conf", 0)):
            fields[field] = {"text": text, "conf": round(conf, 3)}

    print(f"Detected {len(boxes)} boxes")
    print(f"{field} OCR RAW: {text}")
    return fields


# =========================
# Camera & Display
# =========================
def capture(instruction):
    cap = cv2.VideoCapture(0)
    print(f"\n  {instruction}")
    print("  SPACE = capture   Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        bw = int(w * 0.82)
        bh = int(bw / 1.585)
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2

        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x1+bw, y1+bh), (0, 255, 0), 2)
        cv2.putText(display, instruction, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        cv2.putText(display, "Fit license in box — SPACE to capture",
                    (20, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        cv2.imshow("DL Scanner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            captured = frame.copy()
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Quit")

    cap.release()
    cv2.destroyAllWindows()
    return captured


def show_result(title, data, verdict=None):
    canvas = np.ones((450, 720, 3), dtype=np.uint8) * 245
    cv2.putText(canvas, title, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 90, 180), 2)
    cv2.line(canvas, (20, 58), (700, 58), (180, 180, 180), 1)

    y = 105
    for label, value in data.items():
        text = str(value) if value else "NOT FOUND"
        color = (0, 130, 0) if value else (0, 0, 200)
        cv2.putText(canvas, f"{label}:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 80, 80), 1)
        cv2.putText(canvas, text, (180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
        y += 50

    if verdict:
        valid, msg = verdict
        vbg = (215, 255, 215) if valid else (215, 215, 255)
        vcol = (0, 130, 0) if valid else (0, 0, 190)
        cv2.rectangle(canvas, (16, y+4), (704, y+44), vbg, -1)
        cv2.rectangle(canvas, (16, y+4), (704, y+44), vcol, 1)
        cv2.putText(canvas, msg, (24, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, vcol, 2)

    cv2.putText(canvas, "Press any key...", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    cv2.imshow("Result", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# Scan Logic
# =========================
def scan_front():
    best = {"name": None, "dl_no": None, "issue_date": None}

    for attempt in range(1, MAX_RETRIES + 1):
        missing = [f for f, v in best.items() if not v]
        instr = "Show FRONT of Driving License" if attempt == 1 else f"FRONT retry {attempt}/{MAX_RETRIES} — missing: {', '.join(missing)}"

        frame = capture(instr)
        yolo_fields = detect_fields(frame)

        for field in ["name", "dl_no"]:
            if field in yolo_fields and yolo_fields[field]["text"]:
                new_text = yolo_fields[field]["text"]
                new_conf = yolo_fields[field]["conf"]

                if not best[field] or new_conf > 0.85:
                    best[field] = new_text
                    print(f"  ✔ {field:12s}: '{new_text}' (conf={new_conf:.2f})")

        full_texts = extract_all_text(frame)
        issue_date = find_issue_date(full_texts)

        if issue_date and not best["issue_date"]:
            best["issue_date"] = issue_date
            print(f"  ✔ issue_date  : '{issue_date}'")

        if all(best.values()):
            print(f"\n  ✅ All FRONT fields found!")
            break

    return best


def scan_back():
    best_expiry = None

    for attempt in range(1, MAX_RETRIES + 1):
        instr = "Show BACK of Driving License" if attempt == 1 else f"BACK retry {attempt}/{MAX_RETRIES}"

        frame = capture(instr)
        full_texts = extract_all_text(frame)
        expiry_date = find_expiry_date(full_texts)

        if expiry_date and not best_expiry:
            best_expiry = expiry_date
            print(f"  ✔ expiry_date : '{expiry_date}'")
            break

    return best_expiry


# =========================
# Main
# =========================
def main():
    print("=" * 55)
    print("   MULTI-STATE DL SCANNER (Enhanced)")
    print("=" * 55)

    load_yolo()

    print("\n" + "=" * 55)
    print("SCANNING FRONT")
    print("=" * 55)
    front_data = scan_front()

    print("\n" + "=" * 55)
    print("SCANNING BACK")
    print("=" * 55)
    expiry_date = scan_back()

    print("\n" + "=" * 55)
    print("   FINAL RESULT")
    print("=" * 55)
    print(f"  Name       : {front_data['name'] or 'NOT FOUND'}")
    print(f"  DL Number  : {front_data['dl_no'] or 'NOT FOUND'}")
    print(f"  Issue Date : {front_data['issue_date'] or 'NOT FOUND'}")
    print(f"  Expiry Date: {expiry_date or 'NOT FOUND'}")

    verdict = None
    if expiry_date:
        verdict = check_expiry(expiry_date)
        print(f"  Status     : {verdict[1]}")

    print("=" * 55)

    show_result("SCAN RESULT", {
        "Name": front_data["name"],
        "DL Number": front_data["dl_no"],
        "Issue Date": front_data["issue_date"],
        "Expiry Date": expiry_date,
    }, verdict=verdict)

    doc = {
        "name": front_data["name"],
        "dl_number": front_data["dl_no"],
        "issue_date": front_data["issue_date"],
        "expiry_date": expiry_date,
        "valid": verdict[0] if verdict else None,
        "verdict": verdict[1] if verdict else None,
        "created_at": datetime.utcnow(),
    }
    res = collection.insert_one(doc)
    print(f"\n  ✅ Saved to MongoDB (id: {res.inserted_id})")


if __name__ == "__main__":
    main()