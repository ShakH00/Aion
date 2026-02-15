import os
import time
import serial
import cv2
import pytesseract

# ---- CONFIG ----
SERIAL_PORT = "COM3"     # <-- change to your Pico port
BAUD = 115200

# If pytesseract can't find tesseract automatically on Windows, uncomment + fix this path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CAMERA_INDEX = 0
OUT_DIR = "scans"
os.makedirs(OUT_DIR, exist_ok=True)
# ----------------

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    return thresh

def main():
    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not opening. Try CAMERA_INDEX=1 or check permissions.")

    print(f"Connecting to Pico on {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
    time.sleep(2)

    print("Ready. Press Pico button to scan. Ctrl+C to quit.")
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "SCAN":
            print("\nSCAN received → capturing image...")

            # Warmup reads
            for _ in range(5):
                cap.read()
                time.sleep(0.05)

            ok, frame = cap.read()
            if not ok:
                print("Failed to capture frame.")
                continue

            img = preprocess(frame)

            ts = time.strftime("%Y%m%d-%H%M%S")
            img_path = os.path.join(OUT_DIR, f"scan_{ts}.png")
            txt_path = os.path.join(OUT_DIR, f"scan_{ts}.txt")

            cv2.imwrite(img_path, img)
            text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Saved image: {img_path}")
            print(f"Saved text : {txt_path}")
            print("---- OCR TEXT ----")
            print(text.strip())
            print("------------------")

        time.sleep(0.02)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")