"""
dl_camera_app.py — Multi-State Driving License Scanner (Enhanced Accuracy)

Improvements:
  1. Aggressive preprocessing for small text (2x upscale, stronger sharpening)
  2. Multi-OCR fusion (PaddleOCR + EasyOCR + Tesseract)
  3. Smart post-processing (pattern validation, OCR error correction)

Workflow:
  FRONT (3 retries): name, dl_no, issue_date
  BACK (3 retries): expiry_date
  Save to MongoDB
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
import easyocr
import pytesseract
from pymongo import MongoClient

# =========================
# Config
# =========================
YOLO_WEIGHTS = "runs/detect/runs/detect/multistate_dl2/weights/best.pt"
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://corazortechnology:A0Qfk2PbjOMKN32Z@cluster0.drxzj5r.mongodb.net/")

client = MongoClient(MONGO_URI)
collection = client["licenseDB"]["licenses"]

yolo_model = None
paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
easy_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
MAX_RETRIES = 3

YOLO_CLASSES = {0: "name", 1: "dl_no", 2: "dob"}

# =========================
# Enhanced Preprocessing
# =========================
def aggressive_preprocess(crop):
    """Enhanced preprocessing for small/faded text."""
    h, w = crop.shape[:2]

    # 1. Aggressive upscaling (2x minimum)
    if w < 600:
        scale = 600 / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Heavy denoising
    crop = cv2.fastNlMeansDenoisingColored(crop, None, h=15, hColor=15,
                                            templateWindowSize=7, searchWindowSize=21)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE with stronger clip limit
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)

    # 4. Aggressive unsharp masking
    blur = cv2.GaussianBlur(eq, (0, 0), 2.5)
    sharp = cv2.addWeighted(eq, 2.0, blur, -1.0, 0)

    # 5. Adaptive thresholding for faded text
    _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Return both grayscale and binary versions
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR), cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess_full_image(frame):
    """Two-pass preprocessing with aggressive enhancement."""
    h, w = frame.shape[:2]

    # Upscale 2x for small text
    up = cv2.resize(frame, None, fx=2000/w, fy=2000/w, interpolation=cv2.INTER_CUBIC)
    up = cv2.fastNlMeansDenoisingColored(up, None, 15, 15, 7, 21)

    # Pass A — enhanced grayscale
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 3)
    passA = cv2.cvtColor(cv2.addWeighted(eq, 2.0, blur, -1.0, 0), cv2.COLOR_GRAY2BGR)

    # Pass B — red channel isolation
    b, g, r = cv2.split(up)
    red_only = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
    clahe2 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    red_eq = clahe2.apply(red_only)
    blur2 = cv2.GaussianBlur(red_eq, (0, 0), 2.5)
    passB = cv2.cvtColor(cv2.addWeighted(red_eq, 2.2, blur2, -1.2, 0), cv2.COLOR_GRAY2BGR)

    return passA, passB


# =========================
# Multi-OCR Fusion
# =========================
def run_paddle_ocr(img):
    """Run PaddleOCR and return text + confidence."""
    result = paddle_ocr.ocr(img, cls=True)
    texts = []
    for line in result:
        if not line:
            continue
        for word_info in line:
            text = word_info[1][0].strip()
            conf = word_info[1][1]
            if text:
                texts.append((text, conf))
    if not texts:
        return None, 0.0
    # Return joined text with average confidence
    full_text = " ".join([t[0] for t in texts])
    avg_conf = sum([t[1] for t in texts]) / len(texts)
    return full_text, avg_conf


def run_easy_ocr(img):
    """Run EasyOCR and return text + confidence."""
    result = easy_reader.readtext(img)
    if not result:
        return None, 0.0
    texts = [(item[1], item[2]) for item in result]
    full_text = " ".join([t[0] for t in texts])
    avg_conf = sum([t[1] for t in texts]) / len(texts)
    return full_text, avg_conf


def run_tesseract_ocr(img):
    """Run Tesseract OCR and return text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6').strip()
    # Tesseract doesn't return confidence easily, assume 0.7 if text found
    return text if text else None, 0.7 if text else 0.0


def multi_ocr_fusion(crop):
    """Run all 3 OCR engines and pick best result."""
    gray_crop, binary_crop = aggressive_preprocess(crop)

    results = []

    # Run all OCRs on both versions
    for version_name, version_img in [("gray", gray_crop), ("binary", binary_crop)]:
        paddle_text, paddle_conf = run_paddle_ocr(version_img)
        easy_text, easy_conf = run_easy_ocr(version_img)
        tess_text, tess_conf = run_tesseract_ocr(version_img)

        if paddle_text:
            results.append(("Paddle", paddle_text, paddle_conf))
        if easy_text:
            results.append(("Easy", easy_text, easy_conf))
        if tess_text:
            results.append(("Tess", tess_text, tess_conf))

    if not results:
        return None

    # Pick highest confidence result
    results.sort(key=lambda x: x[2], reverse=True)
    best_engine, best_text, best_conf = results[0]

    print(f"    OCR fusion: {best_engine} (conf={best_conf:.2f}) -> '{best_text}'")
    return best_text


# =========================
# Smart Post-Processing
# =========================
def fix_ocr_errors(text):
    """Fix common OCR character substitutions."""
    if not text:
        return text

    # Common OCR errors in alphanumeric strings
    fixes = {
        'O': '0',  # Letter O -> Zero (in digit context)
        'o': '0',
        'I': '1',  # Letter I -> One
        'l': '1',  # Lowercase L -> One
        'S': '5',  # Sometimes S -> 5 in numbers
        'Z': '2',  # Sometimes Z -> 2
    }

    result = list(text)
    for i, ch in enumerate(result):
        prev_digit = i > 0 and result[i-1].isdigit()
        next_digit = i < len(result)-1 and result[i+1].isdigit()

        if ch in fixes and (prev_digit or next_digit):
            result[i] = fixes[ch]

    return ''.join(result)


def validate_dl_number(text):
    """Validate and fix DL number format."""
    if not text:
        return None

    # Remove spaces and fix common errors
    cleaned = fix_ocr_errors(text.replace(" ", "").replace("-", "").upper())

    # Indian DL format: 2 letters + 2-4 digits (RTO) + 4 digits (year) + 7 digits (serial)
    # Total: 15-17 chars, typically 16
    # Example: TS10820200000403 (16 chars)

    # Pattern: XX## #### #######
    if re.match(r'^[A-Z]{2}\d{13,17}$', cleaned):
        return cleaned

    # Try to extract if embedded in longer string
    match = re.search(r'([A-Z]{2}\d{13,17})', cleaned)
    if match:
        return match.group(1)

    return None


def validate_name(text):
    """Validate and clean name."""
    if not text:
        return None

    # Remove non-alphabetic except spaces and hyphens
    cleaned = re.sub(r'[^A-Za-z\s\-]', '', text).strip()

    # Must be at least 2 words, 5 chars total
    words = cleaned.split()
    if len(words) >= 2 and len(cleaned) >= 5:
        return cleaned.upper()

    return None


def validate_date(text):
    """Validate and parse date."""
    if not text:
        return None

    # Extract date patterns
    patterns = [
        r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})',  # DD/MM/YYYY
        r'(\d{8})',                           # DDMMYYYY
        r'(\d{4})/(\d{4})',                   # DDMM/YYYY (garbled)
    ]

    for pat in patterns:
        match = re.search(pat, text)
        if match:
            if len(match.groups()) == 2:  # DDMM/YYYY format
                date_str = match.group(1) + match.group(2)
            else:
                date_str = match.group(1)

            # Try parsing
            for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d%m%Y"]:
                try:
                    dt = datetime.strptime(date_str.replace("/", "").replace("-", "").replace(".", "")
                                          if len(date_str) == 8 else date_str.replace("-", "/").replace(".", "/"),
                                          fmt if "/" in fmt or "-" in fmt or "." in fmt else "%d%m%Y")
                    if 1950 <= dt.year <= 2060:
                        return dt.strftime("%d-%m-%Y")
                except:
                    continue

    return None


# =========================
# OCR + Validation Pipeline
# =========================
def extract_full_image_text(frame):
    """Run multi-OCR on full image."""
    passA, passB = preprocess_full_image(frame)
    all_texts = []
    seen = set()

    for img in [passA, passB]:
        # Run all 3 OCRs
        paddle_text, _ = run_paddle_ocr(img)
        easy_text, _ = run_easy_ocr(img)
        tess_text, _ = run_tesseract_ocr(img)

        for text in [paddle_text, easy_text, tess_text]:
            if not text:
                continue
            key = re.sub(r"\s+", "", text.upper())
            if key not in seen:
                seen.add(key)
                all_texts.append(text)

    return all_texts


def find_issue_date(texts):
    """Find issue date with validation."""
    today = datetime.today()

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
    """Find expiry date with validation."""
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


def detect_yolo_fields(img_bgr):
    """YOLO detection + multi-OCR + validation."""
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
        if field == "dob":
            continue
        if not field:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1-8); y1 = max(0, y1-8)
        x2 = min(w, x2+8); y2 = min(h, y2+8)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Multi-OCR fusion
        text = multi_ocr_fusion(crop)

        # Validate
        if field == "name":
            text = validate_name(text)
        elif field == "dl_no":
            text = validate_dl_number(text)

        if text and (field not in fields or conf > fields[field].get("conf", 0)):
            fields[field] = {"text": text, "conf": round(conf, 3)}

    return fields


# =========================
# Camera + Display (same as before)
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
    best_results = {"name": None, "dl_no": None, "issue_date": None}

    for attempt in range(1, MAX_RETRIES + 1):
        missing = [f for f, v in best_results.items() if not v]
        instr = "Show FRONT of Driving License" if attempt == 1 else f"FRONT retry {attempt}/{MAX_RETRIES} — missing: {', '.join(missing)}"

        frame = capture(instr)
        yolo_fields = detect_yolo_fields(frame)

        for field in ["name", "dl_no"]:
            if field in yolo_fields and yolo_fields[field]["text"]:
                new_text = yolo_fields[field]["text"]
                new_conf = yolo_fields[field]["conf"]

                if not best_results[field] or new_conf > 0.85:
                    best_results[field] = new_text
                    print(f"  ✔ {field:12s}: '{new_text}' (conf={new_conf:.2f})")

        full_texts = extract_full_image_text(frame)
        issue_date = find_issue_date(full_texts)

        if issue_date and not best_results["issue_date"]:
            best_results["issue_date"] = issue_date
            print(f"  ✔ issue_date  : '{issue_date}'")

        if all(best_results.values()):
            print(f"\n  ✅ All FRONT fields found!")
            break

    return best_results


def scan_back():
    best_expiry = None

    for attempt in range(1, MAX_RETRIES + 1):
        instr = "Show BACK of Driving License" if attempt == 1 else f"BACK retry {attempt}/{MAX_RETRIES}"

        frame = capture(instr)
        full_texts = extract_full_image_text(frame)
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
    print("   MULTI-STATE DL SCANNER (Enhanced Accuracy)")
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