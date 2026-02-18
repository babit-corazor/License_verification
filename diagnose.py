"""
diagnose.py — Run PaddleOCR on a saved license image and print raw output.
Helps debug why field parsing is failing without needing to hold up the card.

Usage:
    python diagnose.py                                  # uses default test image
    python diagnose.py WIN_20260217_18_20_21_Pro.jpg    # front
    python diagnose.py WIN_20260217_18_21_31_Pro.jpg    # back
"""
import os, sys, re
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')


def preprocess(frame):
    h, w = frame.shape[:2]
    up = cv2.resize(frame, None, fx=1400/w, fy=1400/w, interpolation=cv2.INTER_CUBIC)
    up = cv2.fastNlMeansDenoisingColored(up, None, 10, 10, 7, 21)

    # Pass A — standard grayscale
    gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (0, 0), 3)
    passA = cv2.cvtColor(cv2.addWeighted(eq, 1.5, blur, -0.5, 0), cv2.COLOR_GRAY2BGR)

    # Pass B — red channel isolation (catches red DL number on Indian licenses)
    b, g, r = cv2.split(up)
    red_only = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
    clahe2   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    red_eq   = clahe2.apply(red_only)
    blur2    = cv2.GaussianBlur(red_eq, (0, 0), 2)
    passB    = cv2.cvtColor(cv2.addWeighted(red_eq, 1.8, blur2, -0.8, 0),
                            cv2.COLOR_GRAY2BGR)
    return passA, passB


def run(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    passA, passB = preprocess(img)

    all_texts = []
    seen = set()

    for label, proc in [("A (grayscale)", passA), ("B (red channel)", passB)]:
        print(f"\n{'='*60}")
        print(f"OCR PASS {label} — {image_path}")
        print(f"{'='*60}")
        result = ocr.ocr(proc, cls=True)
        for line in result:
            if not line:
                continue
            for word_info in line:
                text = word_info[1][0].strip()
                conf = word_info[1][1]
                if not text:
                    continue
                print(f"  [{conf:.2f}]  {text!r}")
                key = re.sub(r'\s+', '', text.upper())
                if key not in seen:
                    seen.add(key)
                    all_texts.append(text)

    print(f"\n{'='*60}")
    print("ALL UNIQUE TOKENS (merged):")
    print(f"{'='*60}")
    for t in all_texts:
        print(f"  {t!r}")

    print(f"\n{'='*60}")
    print("DL NUMBER PATTERN TEST:")
    print(f"{'='*60}")
    full = " ".join(all_texts)

    def normalize(s):
        out = list(s)
        for i, ch in enumerate(out):
            prev_d = i > 0 and out[i-1].isdigit()
            next_d = i < len(out)-1 and out[i+1].isdigit()
            if ch in ('O','o') and (prev_d or next_d): out[i] = '0'
            elif ch in ('I','l') and (prev_d or next_d): out[i] = '1'
        return ''.join(out)

    full_norm = normalize(full)

    patterns = [
        r'[A-Z]{2}\d{2}\s?\d{4}\s?\d{7,8}',
        r'[A-Z]{2}[\s\-]?\d{10,16}',
        r'[A-Z]{2}\d{10,}',
    ]
    for p in patterns:
        m = re.search(p, full_norm)
        print(f"  {p!r:45s} -> {m.group() if m else 'NO MATCH'}")

    print(f"\nTokens that could be DL number fragments:")
    for t in all_texts:
        norm = normalize(t).replace(" ", "")
        if re.search(r'[A-Z]{2}', norm) and re.search(r'\d{4,}', norm):
            print(f"  original={t!r}  normalized={norm!r}")

    print(f"\n{'='*60}")
    print("DATE TOKENS FOUND:")
    print(f"{'='*60}")
    date_pat = re.compile(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}|\d{8}')
    for t in all_texts:
        if date_pat.search(t):
            print(f"  {t!r}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "WIN_20260217_18_20_21_Pro.jpg"
    run(path)