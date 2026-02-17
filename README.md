# Driving License Scanner

A Python application that scans Indian driving licenses using a webcam, extracts key fields using OCR, checks the expiry status, and stores the result in MongoDB.

Built for Telangana state driving licenses (Indian Union Driving Licence format).

---

## Features

- Live webcam capture with a license alignment guide
- Extracts **DL Number**, **Name**, **Issue Date**, and **Expiry Date**
- Automatically checks if the license is **valid or expired**
- **Retry logic** — if a field isn't detected, prompts to re-show the license (up to 3 attempts)
- Dual OCR preprocessing — grayscale pass + red channel isolation to handle red-printed DL numbers
- Stores all extracted data to **MongoDB Atlas**
- Visual result window shows detected fields in green / missing fields in red

---

## How It Works

1. Camera opens — align the **front** of the license inside the guide box and press `SPACE`
2. OCR runs on two preprocessed versions of the image (grayscale + red channel)
3. Extracted: DL Number, Name, Issue Date
4. If any field is missing, prompts to show the card again (up to 3 retries)
5. Camera opens again — align the **back** of the license and press `SPACE`
6. Extracted: Expiry Date → checked against today's date
7. Result displayed on screen and saved to MongoDB

---

## Requirements

- Python 3.10
- Webcam

Install dependencies:

```bash
pip install paddleocr paddlepaddle opencv-python pymongo numpy
```

> On Windows, PaddleOCR may also require:
> ```bash
> pip install shapely pyclipper imgaug
> ```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/driving-license-scanner.git
cd driving-license-scanner
```

**2. Configure MongoDB**

Open `dl_camera_app.py` and replace the `MONGO_URI` with your own MongoDB Atlas connection string:
```python
MONGO_URI = "mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/"
```

**3. Run**
```bash
python dl_camera_app.py
```

---

## Usage

| Key | Action |
|-----|--------|
| `SPACE` | Capture the current frame |
| `Q` | Quit the application |

The app will walk you through scanning the front and back of the license step by step.

---

## Diagnostic Tool

`diagnose.py` runs OCR on a saved image file and prints every detected token — useful for debugging if a field isn't being detected correctly.

```bash
python diagnose.py path/to/license_front.jpg
python diagnose.py path/to/license_back.jpg
```

---

## MongoDB Schema

Each scanned license is stored as a document:

```json
{
  "dl_number":   "TS10820200000403",
  "name":        "BABIT KODALI",
  "issue_date":  "06-01-2020",
  "expiry_date": "05-01-2040",
  "valid":       true,
  "verdict":     "VALID — expires 05 Jan 2040 (5069 days left)",
  "created_at":  "2026-02-17T13:30:00Z"
}
```

---

## Project Structure

```
driving-license-scanner/
├── dl_camera_app.py   # Main application
├── diagnose.py        # OCR debugging tool
└── README.md
```

---

## Known Limitations

- Optimised for Indian (Telangana) driving license format — other state formats may need regex tuning
- Requires good lighting and a steady hand for best OCR accuracy
- Webcam resolution affects detection quality — 720p or higher recommended

---

## Tech Stack

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — OCR engine
- [OpenCV](https://opencv.org/) — image preprocessing and camera
- [MongoDB Atlas](https://www.mongodb.com/atlas) — cloud database
- [NumPy](https://numpy.org/) — image array operations
