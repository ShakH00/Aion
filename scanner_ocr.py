import os
import time
import serial
import cv2
from google import genai
from google.genai import types

SERIAL_PORT = os.getenv("PICO_SERIAL_PORT", "COM9")
BAUD = int(os.getenv("PICO_BAUD", "115200"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
OUT_DIR = os.getenv("SCAN_OUT_DIR", "scans")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.makedirs(OUT_DIR, exist_ok=True)


def _capture_frame(cap):
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to capture image from camera.")
    return frame


def _frame_to_jpeg_bytes(frame_bgr):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode JPEG.")
    return buf.tobytes()


def _gemini_extract_text(image_bytes: bytes) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=GEMINI_API_KEY)

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


def scan_document_with_gemini(timeout=60, save_image=True):
    """
    Blocks until Pico sends 'SCAN' or timeout.
    Returns: (image_bytes, extracted_text, saved_path_or_None)
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not opening. Try CAMERA_INDEX=1 or check permissions.")

    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
    time.sleep(2)

    start = time.time()

    try:
        while True:
            if time.time() - start > timeout:
                raise TimeoutError("No SCAN received before timeout.")

            line = ser.readline().decode(errors="ignore").strip()

            if line == "SCAN":
                frame = _capture_frame(cap)
                image_bytes = _frame_to_jpeg_bytes(frame)

                saved_path = None
                if save_image:
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    saved_path = os.path.join(OUT_DIR, f"scan_{ts}.jpg")
                    cv2.imwrite(saved_path, frame)

                text = _gemini_extract_text(image_bytes)
                return image_bytes, text, saved_path

            time.sleep(0.02)

    finally:
        cap.release()
        ser.close()
