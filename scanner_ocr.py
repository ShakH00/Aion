import os
import time
import serial
import cv2
import numpy as np
import pytesseract

# ---- CONFIG ----
SERIAL_PORT = "COM9"  # <-- change to your Pico port
BAUD = 115200

# If tesseract isn't on PATH, set this on Windows:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CAMERA_INDEX = 0
OUT_DIR = "scans"
os.makedirs(OUT_DIR, exist_ok=True)

# Tesseract settings
TESS_LANG = "eng"
TESS_CONFIG = "--oem 3 --psm 6"
# ----------------

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Remove camera noise but keep edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Normalize lighting across page (background division)
    bg = cv2.medianBlur(gray, 35)
    norm = cv2.divide(gray, bg, scale=255)

    # Gentle contrast boost
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)

    # VERY light adaptive threshold (helps text pop without destroying detail)
    thresh = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    return thresh

def capture_frame(cap) -> np.ndarray:
    # Warm up camera a bit
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to capture image from camera.")
    return frame

def main():
    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try CAMERA_INDEX=1 or check permissions.")

    print(f"Connecting to Pico on {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
    time.sleep(2)

    print("Ready. Press Pico button to scan. Ctrl+C to quit.")

    scan_count = 0
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "SCAN":
            scan_count += 1
            print(f"\n[SCAN #{scan_count}] Capturing image...")

            frame = capture_frame(cap)
            processed = preprocess_for_ocr(frame)

            # OCR on processed image
            text = pytesseract.image_to_string(processed, lang=TESS_LANG, config=TESS_CONFIG)

            ts = time.strftime("%Y%m%d-%H%M%S")
            raw_path = os.path.join(OUT_DIR, f"scan_{ts}_raw.jpg")
            proc_path = os.path.join(OUT_DIR, f"scan_{ts}_processed.png")
            txt_path = os.path.join(OUT_DIR, f"scan_{ts}.txt")

            # Save outputs
            cv2.imwrite(raw_path, frame)
            cv2.imwrite(proc_path, processed)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Saved raw      : {raw_path}")
            print(f"Saved processed: {proc_path}")
            print(f"Saved text     : {txt_path}")
            print("---- OCR TEXT ----")
            print(text.strip())
            print("------------------")

        time.sleep(0.02)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")