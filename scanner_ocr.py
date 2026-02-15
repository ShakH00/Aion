import os
import time
import serial
import cv2

from google import genai
from google.genai import types

# -------- CONFIG --------
SERIAL_PORT = "COM9"   # <-- change to your Pico port
BAUD = 115200
CAMERA_INDEX = 0
OUT_DIR = "scans"
MODEL = "gemini-3-flash-preview"  # vision-capable model in docs
# ------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def capture_frame(cap):
    # Warm up camera a bit
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to capture image from camera.")
    return frame

def frame_to_jpeg_bytes(frame_bgr):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode JPEG.")
    return buf.tobytes()

def gemini_extract_text(client, image_bytes):
    prompt = (
        "Extract ALL text from this image exactly as written. "
        "Preserve line breaks. If something is unclear, keep the best guess "
        "but do not add extra commentary. Output ONLY the extracted text."
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    return (resp.text or "").strip()
def main():
    # Client reads GEMINI_API_KEY from env var
    client = genai.Client(api_key="AIzaSyCm5jSrnB-kXlD6WiS1DtY27RpBeBZLh_g")  # :contentReference[oaicite:2]{index=2}

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

            frame = capture_frame(cap)
            image_bytes = frame_to_jpeg_bytes(frame)

            ts = time.strftime("%Y%m%d-%H%M%S")
            raw_path = os.path.join(OUT_DIR, f"scan_{ts}_raw.jpg")
            txt_path = os.path.join(OUT_DIR, f"scan_{ts}.txt")

            # Save raw image
            cv2.imwrite(raw_path, frame)

            # Gemini OCR
            print("Sending to Gemini...")
            text = gemini_extract_text(client, image_bytes)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Saved image: {raw_path}")
            print(f"Saved text : {txt_path}")
            print("---- TEXT ----")
            print(text)
            print("-------------")

        time.sleep(0.02)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")