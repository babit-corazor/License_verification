"""
dl_scanner_qwen2vl.py — Indian DL Scanner using Qwen2-VL

Uses Qwen2-VL-2B-Instruct for vision-language understanding.
No YOLO needed - the VLM finds fields directly from prompts!
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import cv2
import re
import torch
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pymongo import MongoClient
from PIL import Image
import numpy as np

model = None
processor = None

# =========================
# Load Qwen2-VL Model
# =========================
def load_model():
    global model, processor
    
    print("Loading Qwen2-VL-2B-Instruct (this may take a few minutes)...")
    
    # Load model on GPU
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    print("✅ Qwen2-VL loaded on GPU")


# =========================
# Extract Fields using VLM
# =========================
def extract_dl_info(image_pil, side="front"):
    """Use Qwen2-VL to extract information from license image."""
    
    if side == "front":
        prompt = """Look at this Indian driving license (front side).
Extract the following information:
1. Full name (the person's name, usually below the DL number)
2. DL number (format: 2 letters followed by 13-17 digits, example: TS10820200000403)
3. Issue date (labeled as "Issued On:" or "DOI", format: DD-MM-YYYY or DD/MM/YYYY)

Important: The DL number is printed in RED color at the top.
The person's name is directly below the DL number (NOT the father's name which appears after).

Return ONLY in this exact format:
Name: [full name]
DL Number: [number]
Issue Date: [date]"""
    
    else:  # back
        prompt = """Look at this Indian driving license (back side).
Extract the expiry date/validity date.
It's usually labeled as "Date of Validity" or "Valid Till" or appears after "Non-Transport".

Return ONLY:
Expiry Date: [date in DD-MM-YYYY format]"""
    
    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    
    # Generate
    print(f"  🤖 Processing {side} with Qwen2-VL...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,  # Low temperature for factual extraction
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text


# =========================
# Parse VLM Output
# =========================
def parse_front_output(text):
    """Parse front side VLM output."""
    data = {"name": None, "dl_number": None, "issue_date": None}
    
    # Extract name
    name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if name_match:
        data["name"] = name_match.group(1).strip().upper()
    
    # Extract DL number
    dl_match = re.search(r'DL Number:\s*([A-Z]{2}\d{13,17})', text, re.IGNORECASE)
    if dl_match:
        data["dl_number"] = dl_match.group(1)
    
    # Extract issue date
    date_match = re.search(r'Issue Date:\s*(\d{2}[-/]\d{2}[-/]\d{4})', text, re.IGNORECASE)
    if date_match:
        date_str = date_match.group(1).replace('/', '-')
        data["issue_date"] = date_str
    
    return data


def parse_back_output(text):
    """Parse back side VLM output."""
    # Extract expiry date
    date_match = re.search(r'Expiry Date:\s*(\d{2}[-/]\d{2}[-/]\d{4})', text, re.IGNORECASE)
    if date_match:
        return date_match.group(1).replace('/', '-')
    
    # Fallback: find any date
    date_match = re.search(r'(\d{2}[-/]\d{2}[-/]\d{4})', text)
    if date_match:
        return date_match.group(1).replace('/', '-')
    
    return None


def check_expiry(expiry_str):
    """Check if license is valid."""
    try:
        dt = datetime.strptime(expiry_str, "%d-%m-%Y")
    except:
        try:
            dt = datetime.strptime(expiry_str, "%d/%m/%Y")
        except:
            return None, "Invalid date format"
    
    today = datetime.today()
    if dt < today:
        days = (today - dt).days
        return False, f"EXPIRED — {days} days ago ({dt.strftime('%d %b %Y')})"
    days = (dt - today).days
    return True, f"VALID — expires {dt.strftime('%d %b %Y')} ({days} days left)"


# =========================
# Camera Capture
# =========================
def capture(instruction):
    """Capture image from DroidCam."""
    droidcam_ip = "http://192.168.0.105:4747/video"
    cap = cv2.VideoCapture(droidcam_ip)
    
    if not cap.isOpened():
        print("⚠️  DroidCam not available, using default camera")
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
        cv2.imshow("DL Scanner (Qwen2-VL)", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            captured = frame.copy()
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Quit")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Convert to PIL for Qwen2-VL
    rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    
    return pil_image


# =========================
# Main Scanning Logic
# =========================
def scan_front():
    """Scan front of license."""
    print("\n" + "="*60)
    print("SCANNING FRONT")
    print("="*60)
    
    frame = capture("Show FRONT of Driving License")
    
    # Extract info using Qwen2-VL
    output = extract_dl_info(frame, side="front")
    print(f"\n  📄 VLM Output:\n{output}\n")
    
    # Parse output
    data = parse_front_output(output)
    
    print(f"  ✔ Name       : {data['name'] or 'NOT FOUND'}")
    print(f"  ✔ DL Number  : {data['dl_number'] or 'NOT FOUND'}")
    print(f"  ✔ Issue Date : {data['issue_date'] or 'NOT FOUND'}")
    
    return data


def scan_back():
    """Scan back of license."""
    print("\n" + "="*60)
    print("SCANNING BACK")
    print("="*60)
    
    frame = capture("Show BACK of Driving License")
    
    # Extract expiry date
    output = extract_dl_info(frame, side="back")
    print(f"\n  📄 VLM Output:\n{output}\n")
    
    expiry_date = parse_back_output(output)
    
    print(f"  ✔ Expiry Date: {expiry_date or 'NOT FOUND'}")
    
    return expiry_date


def main():
    print("="*60)
    print("   INDIAN DL SCANNER (Qwen2-VL Edition)")
    print("="*60)
    
    # Load model
    load_model()

    # Connect to MongoDB (moved here so network issues don't block startup)
    MONGO_URI = os.environ.get("MONGO_URI",
                               "")

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        collection = client["licenseDB"]["licenses"]
        # Test connection
        client.server_info()
        print("✅ MongoDB connected")
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("   (Will still scan, just won't save to DB)")
        collection = None
    
    # Scan front
    front_data = scan_front()
    
    # Scan back
    expiry_date = scan_back()
    
    # Final result
    print("\n" + "="*60)
    print("   FINAL RESULT")
    print("="*60)
    print(f"  Name       : {front_data['name'] or 'NOT FOUND'}")
    print(f"  DL Number  : {front_data['dl_number'] or 'NOT FOUND'}")
    print(f"  Issue Date : {front_data['issue_date'] or 'NOT FOUND'}")
    print(f"  Expiry Date: {expiry_date or 'NOT FOUND'}")
    
    verdict = None
    if expiry_date:
        verdict = check_expiry(expiry_date)
        print(f"  Status     : {verdict[1]}")
    
    print("="*60)
    
    # Save to MongoDB
    doc = {
        "name": front_data['name'],
        "dl_number": front_data['dl_number'],
        "issue_date": front_data['issue_date'],
        "expiry_date": expiry_date,
        "valid": verdict[0] if verdict else None,
        "verdict": verdict[1] if verdict else None,
        "scanner_version": "Qwen2-VL",
        "created_at": datetime.utcnow(),
    }
    res = collection.insert_one(doc)
    print(f"\n  ✅ Saved to MongoDB (id: {res.inserted_id})")


if __name__ == "__main__":
    main()
