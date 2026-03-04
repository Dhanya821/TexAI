"""
Cloth Defect Detection Backend - FastAPI
Detects holes, stains, tears in cloth images using YOLOv8
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import base64
import io
import os
import time
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cloth Defect Detection API",
    description="ML-powered API for detecting defects in cloth/fabric images",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loader ───────────────────────────────────────────────────────────

MODEL = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/cloth_defect.pt")

def load_model():
    """Load YOLOv8 model. Falls back to demo mode if model not found."""
    global MODEL
    try:
        from ultralytics import YOLO
        if os.path.exists(MODEL_PATH):
            MODEL = YOLO(MODEL_PATH)
            logger.info("✅ Custom cloth defect model loaded.")
        else:
            # Use pretrained YOLOv8n as placeholder — replace with your trained model
            MODEL = YOLO("yolov8n.pt")
            logger.warning("⚠️  Custom model not found. Using base YOLOv8n (demo mode).")
    except ImportError:
        logger.warning("⚠️  ultralytics not installed. Running in SIMULATION mode.")
        MODEL = None

load_model()

# ─── Image Processing Helpers ───────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to OpenCV image."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def encode_image_base64(img: np.ndarray) -> str:
    """Encode annotated OpenCV image to base64 for frontend display."""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    annotated = img.copy()
    colors = {
        "hole":   (0, 0, 255),      # Red
        "stain":  (0, 165, 255),    # Orange
        "tear":   (0, 255, 255),    # Yellow
        "fray":   (255, 0, 255),    # Magenta
        "defect": (255, 50, 50),    # Blue-ish
    }
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["type"]
        conf  = det["confidence"]
        color = colors.get(label, (0, 255, 0))

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return annotated


# ─── Defect Analysis Logic ──────────────────────────────────────────────────

def run_real_model(img: np.ndarray) -> list:
    """Run actual YOLOv8 inference."""
    results = MODEL(img, conf=0.35, verbose=False)
    detections = []
    class_map = {
        0: "hole", 1: "stain", 2: "tear", 3: "fray"
    }
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = class_map.get(cls_id, "defect")
            detections.append({
                "type": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "area_px": (x2 - x1) * (y2 - y1),
                "severity": "high" if conf > 0.75 else "medium" if conf > 0.5 else "low"
            })
    return detections


def run_simulation(img: np.ndarray) -> list:
    """
    Demo/simulation mode when model is unavailable.
    Runs basic OpenCV-based heuristics to find anomalies.
    """
    h, w = img.shape[:2]
    detections = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Detect dark spots (potential holes) ---
    _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < (h * w * 0.15):
            x, y, bw, bh = cv2.boundingRect(cnt)
            conf = min(0.55 + area / (h * w), 0.95)
            detections.append({
                "type": "hole",
                "confidence": round(conf, 3),
                "bbox": [x, y, x + bw, y + bh],
                "area_px": bw * bh,
                "severity": "high" if conf > 0.75 else "medium"
            })

    # --- Detect color blobs (potential stains) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, stain_mask = cv2.threshold(sat, 120, 255, cv2.THRESH_BINARY)
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel)
    contours2, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if 300 < area < (h * w * 0.20):
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Don't overlap with hole detections
            conf = round(0.50 + area / (h * w * 2), 3)
            conf = min(conf, 0.90)
            detections.append({
                "type": "stain",
                "confidence": conf,
                "bbox": [x, y, x + bw, y + bh],
                "area_px": bw * bh,
                "severity": "medium" if conf > 0.6 else "low"
            })

    # --- Edge-based tear detection ---
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=40, maxLineGap=10)
    if lines is not None and len(lines) > 10:
        xs = [pt for line in lines for pt in [line[0][0], line[0][2]]]
        ys = [pt for line in lines for pt in [line[0][1], line[0][3]]]
        if xs and ys:
            x1, y1 = max(0, min(xs) - 5), max(0, min(ys) - 5)
            x2, y2 = min(w, max(xs) + 5), min(h, max(ys) + 5)
            conf = min(0.45 + len(lines) * 0.005, 0.85)
            detections.append({
                "type": "tear",
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "area_px": (x2 - x1) * (y2 - y1),
                "severity": "high" if conf > 0.70 else "low"
            })

    # Limit detections
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)[:6]
    return detections


def analyze_cloth(img: np.ndarray) -> dict:
    """Main analysis function — uses real model or simulation."""
    start = time.time()

    if MODEL is not None:
        try:
            detections = run_real_model(img)
        except Exception as e:
            logger.warning(f"Model inference failed: {e}. Falling back to simulation.")
            detections = run_simulation(img)
    else:
        detections = run_simulation(img)

    elapsed = round(time.time() - start, 3)

    # Summary stats
    type_counts = {}
    for d in detections:
        type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1

    summary_parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in type_counts.items()]
    summary = f"{len(detections)} defect(s) found: {', '.join(summary_parts)}" \
              if detections else "No defects detected — cloth appears clean ✅"

    overall_condition = "good"
    if len(detections) == 0:
        overall_condition = "good"
    elif len(detections) <= 2:
        overall_condition = "minor_damage"
    elif len(detections) <= 4:
        overall_condition = "moderate_damage"
    else:
        overall_condition = "severe_damage"

    return {
        "detections": detections,
        "summary": summary,
        "total_defects": len(detections),
        "defect_types": type_counts,
        "overall_condition": overall_condition,
        "processing_time_sec": elapsed,
        "mode": "model" if MODEL is not None else "simulation"
    }


# ─── API Routes ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Cloth Defect Detection API is running 🧵", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else "not found (demo mode)"
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload a cloth image and receive defect analysis results.
    Returns: detections list, annotated image (base64), summary stats.
    """
    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type: {file.content_type}. Use JPG/PNG.")

    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

    try:
        img = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Run analysis
    result = analyze_cloth(img)

    # Draw annotations on image
    annotated_img = draw_detections(img, result["detections"])
    annotated_b64 = encode_image_base64(annotated_img)
    original_b64  = encode_image_base64(img)

    return JSONResponse({
        "status": "success",
        "filename": file.filename,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "analysis": result,
        "annotated_image": annotated_b64,
        "original_image": original_b64
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)