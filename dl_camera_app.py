"""
dl_camera_app.py — Multi-State Driving License Scanner

Workflow:
  FRONT (3 retries): Detect name, dl_no, issue_date
  BACK  (3 retries): Detect expiry_date
  Save to MongoDB

Fields are retained across retries — only overwritten if better confidence/result found.
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

YOLO_CLASSES = {
    0: "name",
    1: "dl_no",
    2: "dob",  # detected but not used
}

# =========================
# Preprocessing
# =========================
def preprocess_for_ocr(crop):
    """Enhance cropped field before OCR."""
    h, w = crop.shape[:2]
    if w < 300:
        scale = 300 / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crop = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 3)
    sharp = cv2.addWeighted(eq, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def preprocess_full_image(frame):
    """Two-pass preprocessing for full image OCR."""
    h, w = frame.shape[:2]
    up = cv2.resize(frame, None, fx=1400/w, fy=1400/w, interpolation=cv2.INTER_CUBIC)
    up = cv2.fastNlMeansDenoisingColored(up, None, 10, 10, 7, 21)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 3)
    passA = cv2.cvtColor(cv2.addWeighted(eq, 1.5, blur, -0.5, 0), cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(up)
    red_only = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    red_eq = clahe2.apply(red_only)
    blur2 = cv2.GaussianBlur(red_eq, (0, 0), 2)
    passB = cv2.cvtColor(cv2.addWeighted(red_eq, 1.8, blur2, -0.8, 0), cv2.COLOR_GRAY2BGR)

    return passA, passB


# =========================
# OCR
# =========================
def read_text_from_crop(crop_bgr):
    """Run PaddleOCR on single crop."""
    enhanced = preprocess_for_ocr(crop_bgr)
    result = ocr.ocr(enhanced, cls=True)
    texts = []
    for line in result:
        if not line:
            continue
        for word_info in line:
            text = word_info[1][0].strip()
            if text:
                texts.append(text)
    return " ".join(texts) if texts else None


def extract_full_image_text(frame):
    """Run OCR on full image (both passes)."""
    passA, passB = preprocess_full_image(frame)
    all_texts, seen = [], set()

    for label, img in [("A", passA), ("B-red", passB)]:
        result = ocr.ocr(img, cls=True)
        for line in result:
            if not line:
                continue
            for word_info in line:
                text = word_info[1][0].strip()
                if not text:
                    continue
                key = re.sub(r"\s+", "", text.upper())
                if key not in seen:
                    seen.add(key)
                    all_texts.append(text)
    return all_texts


# =========================
# Date Parsing
# =========================
DATE_FMTS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d%m%Y",
             "%d %b %Y", "%d %B %Y", "%m/%d/%Y"]

def try_parse_date(s):
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_dates_from_token(text):
    found = []
    for m in re.findall(r"\d{1,2}[/\-\.]\d{2}[/\-\.]\d{4}", text):
        dt = try_parse_date(m)
        if dt and 1950 <= dt.year <= 2060:
            found.append(dt)
    for m in re.findall(r"(\d{4})/(\d{4})", text):
        dt = try_parse_date(m[0] + m[1])
        if dt and 1950 <= dt.year <= 2060:
            found.append(dt)
    for m in re.findall(r"\b(\d{8})\b", text):
        dt = try_parse_date(m)
        if dt and 1950 <= dt.year <= 2060:
            found.append(dt)
    return found


def find_all_dates(texts):
    found = []
    for text in texts:
        cleaned = re.sub(
            r"(issued?\s*on|date\s*of\s*(validity|issue|birth)|dob)\s*[:\-]?\s*",
            "", text, flags=re.IGNORECASE
        ).strip()
        for dt in extract_dates_from_token(cleaned):
            found.append((dt, text))
    return found


def parse_issue_date(texts):
    today = datetime.today()
    # Priority: look for "issued" or "on" label
    for text in texts:
        if re.search(r"(?:on|iss)", text, re.IGNORECASE):
            for dt in extract_dates_from_token(text):
                if 2000 <= dt.year <= today.year:
                    return dt.strftime("%d-%m-%Y")

    # Fallback: earliest date between 2000-today
    candidates = [(dt, r) for dt, r in find_all_dates(texts)
                  if 2000 <= dt.year <= today.year]
    if candidates:
        candidates.sort()
        return candidates[0][0].strftime("%d-%m-%Y")
    return None


def parse_expiry_date(texts):
    combined = " ".join(texts)
    # Priority: explicit validity label
    m = re.search(
        r"(?:date\s*of\s*validity|valid\s*(?:till|upto|up\s*to)|expiry)\s*[:\-]?\s*"
        r"(\d{1,2}[/\-\.]\d{2}[/\-\.]\d{4})",
        combined, flags=re.IGNORECASE)
    if m:
        dt = try_parse_date(m.group(1))
        if dt and dt.year >= 2025:
            return dt.strftime("%d-%m-%Y")

    # Fallback: any future date >= 2025
    future = [(dt, r) for dt, r in find_all_dates(texts) if dt.year >= 2025]
    if future:
        future.sort()
        return future[0][0].strftime("%d-%m-%Y")
    return None


def check_expiry(expiry_str):
    dt = datetime.strptime(expiry_str, "%d-%m-%Y")
    today = datetime.today()
    if dt < today:
        days = (today - dt).days
        return False, f"EXPIRED {days} days ago ({dt.strftime('%d %b %Y')})"
    days = (dt - today).days
    return True, f"VALID expires {dt.strftime('%d %b %Y')} ({days} days left)"


# =========================
# YOLO + OCR
# =========================
def load_yolo():
    global yolo_model
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(
            f"YOLO weights not found: {YOLO_WEIGHTS}\n"
            "Run training first: python train.py"
        )
    device = 0 if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(YOLO_WEIGHTS)
    print(f"  ✅ YOLO loaded (device={device})")


def detect_yolo_fields(img_bgr):
    """Run YOLO detection and return name + dl_no (skip dob)."""
    h, w = img_bgr.shape[:2]
    if w < 800:
        scale = 800 / w
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]

    device = 0 if torch.cuda.is_available() else "cpu"
    results = yolo_model.predict(img_bgr, conf=0.25, device=device, verbose=False)
    boxes = results[0].boxes

    fields = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        field = YOLO_CLASSES.get(cls_id)
        if field == "dob":  # skip dob
            continue
        if not field:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1-5); y1 = max(0, y1-5)
        x2 = min(w, x2+5); y2 = min(h, y2+5)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text = read_text_from_crop(crop)

        # Keep highest confidence result
        if field not in fields or conf > fields[field].get("conf", 0):
            fields[field] = {"text": text, "conf": round(conf, 3)}

    return fields


# =========================
# Camera
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


# =========================
# Result Display
# =========================
def show_result(title, data, verdict=None):
    canvas = np.ones((450, 720, 3), dtype=np.uint8) * 245
    cv2.putText(canvas, title, (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 90, 180), 2)
    cv2.line(canvas, (20, 58), (700, 58), (180, 180, 180), 1)

    y = 105
    for label, value in data.items():
        text = str(value) if value else "NOT FOUND"
        color = (0, 130, 0) if value else (0, 0, 200)
        cv2.putText(canvas, f"{label}:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 80, 80), 1)
        cv2.putText(canvas, text, (180, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
        y += 50

    if verdict:
        valid, msg = verdict
        vbg = (215, 255, 215) if valid else (215, 215, 255)
        vcol = (0, 130, 0) if valid else (0, 0, 190)
        cv2.rectangle(canvas, (16, y+4), (704, y+44), vbg, -1)
        cv2.rectangle(canvas, (16, y+4), (704, y+44), vcol, 1)
        cv2.putText(canvas, msg, (24, y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, vcol, 2)

    cv2.putText(canvas, "Press any key...", (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    cv2.imshow("Result", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# Scan Logic (Front/Back with retries + field retention)
# =========================
def scan_front():
    """Scan FRONT — extract name, dl_no, issue_date with retry logic."""
    best_results = {
        "name": None,
        "dl_no": None,
        "issue_date": None,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        missing = [f for f, v in best_results.items() if not v]

        if attempt == 1:
            instr = "Show FRONT of Driving License"
        else:
            instr = f"FRONT retry {attempt}/{MAX_RETRIES} — missing: {', '.join(missing)}"

        frame = capture(instr)

        # YOLO detection
        yolo_fields = detect_yolo_fields(frame)

        # Update name and dl_no if better result found
        for field in ["name", "dl_no"]:
            if field in yolo_fields and yolo_fields[field]["text"]:
                new_text = yolo_fields[field]["text"]
                new_conf = yolo_fields[field]["conf"]

                # Keep if: (1) we don't have it yet, OR (2) new confidence is higher
                if not best_results[field]:
                    best_results[field] = new_text
                    print(f"  ✔ {field:12s}: '{new_text}' (conf={new_conf:.2f})")
                elif new_conf > 0.9:  # very high confidence — trust it
                    best_results[field] = new_text
                    print(f"  ✔ {field:12s}: '{new_text}' (conf={new_conf:.2f}, updated)")
                else:
                    print(f"  → {field:12s}: kept previous (new conf={new_conf:.2f} not better)")

        # Full-image OCR for issue_date
        full_texts = extract_full_image_text(frame)
        issue_date = parse_issue_date(full_texts)

        # Update issue_date if found or better
        if issue_date and not best_results["issue_date"]:
            best_results["issue_date"] = issue_date
            print(f"  ✔ issue_date  : '{issue_date}'")
        elif issue_date:
            print(f"  → issue_date  : kept previous ('{best_results['issue_date']}')")
        else:
            print(f"  ✗ issue_date  : not found")

        # Check if all found
        if all(best_results.values()):
            print(f"\n  ✅ All FRONT fields found!")
            break

        if attempt == MAX_RETRIES:
            print(f"\n  ⚠️ Max retries — continuing with available data")

    return best_results


def scan_back():
    """Scan BACK — extract expiry_date with retry logic."""
    best_expiry = None

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt == 1:
            instr = "Show BACK of Driving License"
        else:
            instr = f"BACK retry {attempt}/{MAX_RETRIES} — expiry not found"

        frame = capture(instr)
        full_texts = extract_full_image_text(frame)
        expiry_date = parse_expiry_date(full_texts)

        if expiry_date and not best_expiry:
            best_expiry = expiry_date
            print(f"  ✔ expiry_date : '{expiry_date}'")
            break
        elif expiry_date:
            print(f"  → expiry_date : kept previous ('{best_expiry}')")
            break
        else:
            print(f"  ✗ expiry_date : not found")

        if attempt == MAX_RETRIES:
            print(f"\n  ⚠️ Max retries — continuing with available data")

    return best_expiry


# =========================
# Main
# =========================
def main():
    print("=" * 55)
    print("   MULTI-STATE DRIVING LICENSE SCANNER")
    print("=" * 55)

    load_yolo()

    # Scan FRONT
    print("\n" + "=" * 55)
    print("SCANNING FRONT")
    print("=" * 55)
    front_data = scan_front()

    # Scan BACK
    print("\n" + "=" * 55)
    print("SCANNING BACK")
    print("=" * 55)
    expiry_date = scan_back()

    # Final summary
    print("\n" + "=" * 55)
    print("   FINAL RESULT")
    print("=" * 55)
    print(f"  Name       : {front_data['name'] or 'NOT FOUND'}")
    print(f"  DL Number  : {front_data['dl_no'] or 'NOT FOUND'}")
    print(f"  Issue Date : {front_data['issue_date'] or 'NOT FOUND'}")
    print(f"  Expiry Date: {expiry_date or 'NOT FOUND'}")

    # Expiry check
    verdict = None
    if expiry_date:
        verdict = check_expiry(expiry_date)
        print(f"  Status     : {verdict[1]}")

    print("=" * 55)

    # Show result window
    show_result("SCAN RESULT", {
        "Name": front_data["name"],
        "DL Number": front_data["dl_no"],
        "Issue Date": front_data["issue_date"],
        "Expiry Date": expiry_date,
    }, verdict=verdict)

    # Save to MongoDB
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