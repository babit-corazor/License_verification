import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import re
import numpy as np
from datetime import datetime
from paddleocr import PaddleOCR
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

# =========================
# MongoDB
# =========================
MONGO_URI = os.environ.get("MONGO_URI")
client     = MongoClient(MONGO_URI)
collection = client["licenseDB"]["licenses"]

ocr         = PaddleOCR(use_angle_cls=True, lang="en")
MAX_RETRIES = 3

# =========================
# Preprocessing
# =========================
def preprocess(frame):
    h, w = frame.shape[:2]
    up = cv2.resize(frame, None, fx=1400/w, fy=1400/w, interpolation=cv2.INTER_CUBIC)
    up = cv2.fastNlMeansDenoisingColored(up, None, 10, 10, 7, 21)

    # Pass A — grayscale + CLAHE + sharpen
    gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (0, 0), 3)
    passA = cv2.cvtColor(cv2.addWeighted(eq, 1.5, blur, -0.5, 0), cv2.COLOR_GRAY2BGR)

    # Pass B — red channel isolation (for red-printed DL numbers on Indian licenses)
    b, g, r  = cv2.split(up)
    red_only = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
    clahe2   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    red_eq   = clahe2.apply(red_only)
    blur2    = cv2.GaussianBlur(red_eq, (0, 0), 2)
    passB    = cv2.cvtColor(cv2.addWeighted(red_eq, 1.8, blur2, -0.8, 0), cv2.COLOR_GRAY2BGR)

    return passA, passB

# =========================
# OCR
# =========================
def extract_text(frame):
    passA, passB = preprocess(frame)
    all_texts, all_confs, seen = [], [], set()
    for label, img in [("A", passA), ("B-red", passB)]:
        result = ocr.ocr(img, cls=True)
        print(f"\n--- OCR pass {label} ---")
        for line in result:
            if not line:
                continue
            for word_info in line:
                text = word_info[1][0].strip()
                conf = word_info[1][1]
                if not text:
                    continue
                print(f"  [{conf:.2f}] {text!r}")
                key = re.sub(r"\s+", "", text.upper())
                if key not in seen:
                    seen.add(key)
                    all_texts.append(text)
                    all_confs.append(conf)
    print(f"--- {len(all_texts)} unique tokens ---")
    return all_texts, all_confs

# =========================
# Normalise OCR errors
# =========================
def normalize_ocr(text):
    """Replace O->0 and I/l->1 only when adjacent to digit characters."""
    out = list(text)
    for i, ch in enumerate(out):
        prev_d = i > 0 and out[i-1].isdigit()
        next_d = i < len(out)-1 and out[i+1].isdigit()
        if ch in ("O", "o") and (prev_d or next_d):
            out[i] = "0"
        elif ch in ("I", "l") and (prev_d or next_d):
            out[i] = "1"
    return "".join(out)

# =========================
# DL Number Parser
#
# FIX: diagnose.py showed OCR reads TS1082020000040 (15 digits, last char
# dropped) instead of TS10820200000403 (16 digits). Changed minimum from
# 10 to 13 so 15-digit result still matches. Also added 3-token join fallback.
# =========================
def parse_dl_number(texts):
    # 1. Single token — 2 letters + 13-16 digits
    for text in texts:
        norm = normalize_ocr(text).replace(" ", "").replace("-", "")
        if re.match(r"^[A-Z]{2}\d{13,16}$", norm):
            return norm

    # 2. Search full joined string
    full_norm = normalize_ocr(" ".join(texts))
    for pat in [r"([A-Z]{2}\d{2}\s?\d{4}\s?\d{6,8})",
                r"([A-Z]{2}[\s\-]?\d{13,16})"]:
        m = re.search(pat, full_norm)
        if m:
            return m.group(1).replace(" ", "").replace("-", "")

    # 3. Two adjacent tokens joined
    for i in range(len(texts) - 1):
        combo = normalize_ocr(texts[i] + texts[i+1]).replace(" ", "")
        if re.match(r"^[A-Z]{2}\d{13,16}$", combo):
            return combo

    # 4. Three adjacent tokens joined (DL split across 3 OCR regions)
    for i in range(len(texts) - 2):
        combo = normalize_ocr(texts[i] + texts[i+1] + texts[i+2]).replace(" ", "")
        if re.match(r"^[A-Z]{2}\d{13,16}$", combo):
            return combo

    return None

# =========================
# Name Parser
# =========================
BLACKLIST = {
    "DRIVING","LICENCE","LICENSE","TRANSPORT","INDIA","GOVT","VALID","DOB",
    "DATE","TELANGANA","STATE","UNION","INDIAN","ISSUED","ISSUE","EXPIRY",
    "BIRTH","AUTHORITY","LICENSING","RTA","UPPAL","KAPRA","NAGAR","MEDCHAL",
    "PLOT","BEHIND","MARKET","SIGNATURE","NON","MOTOR","CYCLE","VEHICLE",
    "LIGHT","GEAR","WITH","VALIDITY","REFERENCE","ORIGINAL","BADGE","BLOOD",
    "GROUP","CLASS","DOV","COV","SHOW","FRONT","BACK","AGAIN","RETRY",
    "ALIGN","PRESS","SPACE","SCANNER","FIT","INSIDE","BOX",
}

def is_name(text):
    words = text.strip().split()
    upper = text.upper()
    return (
        len(words) >= 2
        and len(text) >= 5
        and text.replace(" ", "").replace("-", "").isalpha()
        and not any(bw in upper.split() for bw in BLACKLIST)
        and not re.search(r"\d", text)
    )

def parse_name(texts):
    # Strategy 1: line right after the DL number token
    dl_idx = None
    for i, text in enumerate(texts):
        norm = normalize_ocr(text).replace(" ", "")
        if re.match(r"^[A-Z]{2}\d{13,16}$", norm):
            dl_idx = i
            break
    if dl_idx is not None:
        for j in range(dl_idx + 1, min(dl_idx + 5, len(texts))):
            if is_name(texts[j]):
                return texts[j].strip()

    # Strategy 2: ALL-CAPS 2+ word alpha line (Indian DLs print name in caps)
    for text in texts:
        if is_name(text) and text == text.upper():
            return text.strip()

    # Strategy 3: any clean name-like line
    for text in texts:
        if is_name(text):
            return text.strip()

    return None

# =========================
# Date Parsers
#
# FIX: diagnose.py showed "Issued On: 06/01/2020" is read as "ludOn:0601/2020"
# The garbled format DDMM/YYYY (0601/2020) is now handled by extract_dates_from_token.
# =========================
DATE_FMTS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d%m%Y", "%d %b %Y", "%d %B %Y", "%m/%d/%Y",
]

def try_parse_date(s):
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None

def extract_dates_from_token(text):
    """
    Aggressively extract dates from a potentially garbled OCR token.
    Handles: dd/mm/yyyy, dd-mm-yyyy, ddmmyyyy, and the garbled ddmm/yyyy form.
    """
    found = []

    # Standard format: 06/01/2020 or 06-01-2020
    for m in re.findall(r"\d{1,2}[/\-\.]\d{2}[/\-\.]\d{4}", text):
        dt = try_parse_date(m)
        if dt and 1950 <= dt.year <= 2060:
            found.append(dt)

    # Garbled format: 0601/2020 -> DDMM/YYYY -> join as DDMMYYYY
    for m in re.findall(r"(\d{4})/(\d{4})", text):
        dt = try_parse_date(m[0] + m[1])
        if dt and 1950 <= dt.year <= 2060:
            found.append(dt)

    # Pure 8-digit run: 06012020
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
    # FIX: search for any token containing "on" or "iss" with an embedded date
    # This catches "ludOn:0601/2020" which is a garbled "Issued On: 06/01/2020"
    today = datetime.today()
    for text in texts:
        if re.search(r"(?:on|iss)", text, re.IGNORECASE):
            for dt in extract_dates_from_token(text):
                if 2000 <= dt.year <= today.year:
                    return dt.strftime("%d-%m-%Y")

    # Fallback: earliest date between 2000 and today
    candidates = [(dt, r) for dt, r in find_all_dates(texts)
                  if 2000 <= dt.year <= today.year]
    if candidates:
        candidates.sort()
        return candidates[0][0].strftime("%d-%m-%Y")
    return None

def parse_expiry_date(texts):
    combined = " ".join(texts)
    m = re.search(
        r"(?:date\s*of\s*validity|valid\s*(?:till|upto|up\s*to)|expiry)\s*[:\-]?\s*"
        r"(\d{1,2}[/\-\.]\d{2}[/\-\.]\d{4})",
        combined, flags=re.IGNORECASE)
    if m:
        dt = try_parse_date(m.group(1))
        if dt and dt.year >= 2025:
            return dt.strftime("%d-%m-%Y")
    future = [(dt, r) for dt, r in find_all_dates(texts) if dt.year >= 2025]
    if future:
        future.sort()
        return future[0][0].strftime("%d-%m-%Y")
    return None

# =========================
# Expiry Check
# =========================
def check_expiry(expiry_str):
    dt    = datetime.strptime(expiry_str, "%d-%m-%Y")
    today = datetime.today()
    if dt < today:
        days = (today - dt).days
        return False, f"EXPIRED — {days} days ago ({dt.strftime('%d %b %Y')})"
    days = (dt - today).days
    return True, f"VALID — expires {dt.strftime('%d %b %Y')} ({days} days left)"

# =========================
# Camera Capture
# KEY FIX: all overlay drawn on display copy — OCR runs on the clean frame.
# Previous bug: instruction text was drawn onto frame itself, causing OCR to
# read "Show FRONT again" as part of the license text.
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

        display = frame.copy()   # draw on copy only
        cv2.rectangle(display, (x1, y1), (x1+bw, y1+bh), (0, 255, 0), 2)
        cv2.putText(display, instruction, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        cv2.putText(display, "Fit license in box — SPACE to capture",
                    (20, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        cv2.imshow("DL Scanner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            captured = frame.copy()   # clean frame — no text overlay
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
    canvas = np.ones((440, 720, 3), dtype=np.uint8) * 245
    cv2.putText(canvas, title, (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 90, 180), 2)
    cv2.line(canvas, (20, 58), (700, 58), (180, 180, 180), 1)
    y = 105
    for label, value in data.items():
        text  = str(value) if value else "NOT FOUND"
        color = (0, 130, 0) if value else (0, 0, 200)
        cv2.putText(canvas, f"{label}:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 80, 80), 1)
        cv2.putText(canvas, text, (210, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
        y += 50
    if verdict:
        valid, msg = verdict
        vbg  = (215, 255, 215) if valid else (215, 215, 255)
        vcol = (0, 130, 0)     if valid else (0, 0, 190)
        cv2.rectangle(canvas, (16, y+4), (704, y+44), vbg, -1)
        cv2.rectangle(canvas, (16, y+4), (704, y+44), vcol, 1)
        cv2.putText(canvas, msg, (24, y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, vcol, 2)
    cv2.putText(canvas, "Press any key...", (20, 425),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    cv2.imshow("Result", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# Scan with Retry
# =========================
FRONT_TIPS = {
    "DL Number":  "Tip: DL number at top must be fully visible and in focus",
    "Name":       "Tip: hold card flat — name line must be sharp",
    "Issue Date": "Tip: bottom of card — Issued On date must be visible",
}

def scan_front():
    dl_number = name = issue_date = None
    missing = ["DL Number", "Name", "Issue Date"]

    for attempt in range(1, MAX_RETRIES + 1):
        instr = ("Show FRONT of Driving License" if attempt == 1
                 else f"FRONT retry {attempt}/{MAX_RETRIES} — missing: {', '.join(missing)}")
        frame  = capture(instr)
        texts, _ = extract_text(frame)

        dl_number  = parse_dl_number(texts)
        name       = parse_name(texts)
        issue_date = parse_issue_date(texts)

        missing = [f for f, v in [
            ("DL Number", dl_number),
            ("Name", name),
            ("Issue Date", issue_date)
        ] if not v]

        print(f"\n  Attempt {attempt}:")
        print(f"    DL Number : {dl_number  or 'NOT FOUND'}")
        print(f"    Name      : {name       or 'NOT FOUND'}")
        print(f"    Issue Date: {issue_date or 'NOT FOUND'}")

        if not missing:
            print("    All found!")
            break
        if attempt < MAX_RETRIES:
            for f in missing:
                print(f"    {FRONT_TIPS.get(f, '')}")
        else:
            print("    Max retries — continuing with partial data")

    show_result("FRONT SIDE", {
        "DL Number":  dl_number,
        "Name":       name,
        "Issue Date": issue_date,
    })
    return {"dl_number": dl_number, "name": name, "issue_date": issue_date}


def scan_back():
    expiry_date = None
    for attempt in range(1, MAX_RETRIES + 1):
        instr = ("Show BACK of Driving License" if attempt == 1
                 else f"BACK retry {attempt}/{MAX_RETRIES} — expiry not found")
        frame = capture(instr)
        texts, _ = extract_text(frame)
        expiry_date = parse_expiry_date(texts)

        print(f"\n  Attempt {attempt}: Expiry = {expiry_date or 'NOT FOUND'}")
        if expiry_date:
            print("    Found!")
            break
        if attempt < MAX_RETRIES:
            print("    Tip: ensure Date of Validity row is clearly visible")
        else:
            print("    Max retries — continuing")

    verdict = check_expiry(expiry_date) if expiry_date else None
    show_result("BACK SIDE", {"Expiry Date": expiry_date}, verdict=verdict)
    return expiry_date, verdict

# =========================
# Main
# =========================
def main():
    print("=" * 55)
    print("   DRIVING LICENSE SCANNER")
    print("=" * 55)

    front_data           = scan_front()
    expiry_date, verdict = scan_back()

    print("\n" + "=" * 55)
    print("   FINAL RESULT")
    print("=" * 55)
    print(f"  DL Number  : {front_data['dl_number']  or 'NOT FOUND'}")
    print(f"  Name       : {front_data['name']       or 'NOT FOUND'}")
    print(f"  Issue Date : {front_data['issue_date'] or 'NOT FOUND'}")
    print(f"  Expiry Date: {expiry_date              or 'NOT FOUND'}")
    if verdict:
        valid, msg = verdict
        print(f"  Status     : {'OK' if valid else 'FAIL'} — {msg}")
    print("=" * 55)

    doc = {
        "dl_number":   front_data["dl_number"],
        "name":        front_data["name"],
        "issue_date":  front_data["issue_date"],
        "expiry_date": expiry_date,
        "valid":       verdict[0] if verdict else None,
        "verdict":     verdict[1] if verdict else None,
        "created_at":  datetime.utcnow(),
    }
    res = collection.insert_one(doc)
    print(f"\n  Saved to MongoDB (id: {res.inserted_id})")


if __name__ == "__main__":
    main()