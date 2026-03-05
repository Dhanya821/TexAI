"""
Cloth Defect Detection Backend - FastAPI
Detects holes, overthreads, strains, tears, uneven colours in cloth images
using Roboflow inference_sdk workflow.
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
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cloth Defect Detection API",
    description="ML-powered API for detecting defects in cloth/fabric images via Roboflow",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Roboflow Client Setup ───────────────────────────────────────────────────

ROBOFLOW_API_KEY    = "HMI2IUb0vkpOuClaVqOx"
ROBOFLOW_WORKSPACE  = "yugass-workspace"
ROBOFLOW_WORKFLOW   = "custom-workflow"

CLIENT = None

def load_client():
    global CLIENT
    try:
        from inference_sdk import InferenceHTTPClient
        CLIENT = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        logger.info("✅ Roboflow InferenceHTTPClient initialised.")
    except ImportError:
        logger.warning("⚠️  inference_sdk not installed. Running in SIMULATION mode.")
        logger.warning("    Install via: pip install inference-sdk")
        CLIENT = None

load_client()

# ─── Image Processing Helpers ────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def encode_image_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    annotated = img.copy()
    colors = {
        "hole":          (0,   0,   255),   # Red
        "overthread":    (0,   165, 255),   # Orange
        "strain":        (0,   255, 255),   # Yellow
        "tear":          (255, 0,   255),   # Magenta
        "unevencolours": (255, 128, 0  ),   # Blue-ish
        "defect":        (50,  205, 50 ),   # Green fallback
    }
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["type"]
        conf  = det["confidence"]
        color = colors.get(label, (0, 255, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return annotated


# ─── Roboflow Workflow Inference ─────────────────────────────────────────────

def parse_roboflow_result(raw_result: list) -> list:
    """
    Parse the workflow result from Roboflow into our standard detection format.
    Roboflow workflows typically return a list with one dict per image.
    Prediction keys vary by workflow — we handle common structures gracefully.
    """
    detections = []

    if not raw_result:
        return detections

    # raw_result is a list; each element corresponds to one image
    item = raw_result[0] if isinstance(raw_result, list) else raw_result

    # Locate the predictions list — workflow outputs vary
    predictions = None
    if isinstance(item, dict):
        # Common keys: 'predictions', 'output', 'outputs'
        for key in ("predictions", "output", "outputs", "result"):
            val = item.get(key)
            if val is not None:
                if isinstance(val, dict) and "predictions" in val:
                    predictions = val["predictions"]
                elif isinstance(val, list):
                    predictions = val
                break
        # Also handle nested: item['predictions']['predictions']
        if predictions is None and "predictions" in item:
            nested = item["predictions"]
            if isinstance(nested, dict) and "predictions" in nested:
                predictions = nested["predictions"]
            elif isinstance(nested, list):
                predictions = nested

    if not predictions:
        logger.warning("No predictions found in Roboflow response: %s", item)
        return detections

    for pred in predictions:
        # Roboflow object-detection format
        x_center = pred.get("x", 0)
        y_center = pred.get("y", 0)
        width    = pred.get("width",  0)
        height   = pred.get("height", 0)
        conf     = float(pred.get("confidence", 0))
        label    = str(pred.get("class", "defect")).lower().replace(" ", "")

        x1 = int(x_center - width  / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width  / 2)
        y2 = int(y_center + height / 2)

        detections.append({
            "type":       label,
            "confidence": round(conf, 3),
            "bbox":       [x1, y1, x2, y2],
            "area_px":    int(width * height),
            "severity":   "high" if conf > 0.75 else "medium" if conf > 0.5 else "low"
        })

    return detections


def run_roboflow(image_bytes: bytes) -> list:
    """Send image to Roboflow workflow and return parsed detections."""
    # Save to a temp file — inference_sdk accepts file paths
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        raw = CLIENT.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW,
            images={"image": tmp_path},
            use_cache=True
        )
        logger.info("Roboflow raw response: %s", raw)
        return parse_roboflow_result(raw)
    finally:
        os.unlink(tmp_path)


# ─── Simulation Fallback ─────────────────────────────────────────────────────

def run_simulation(img: np.ndarray) -> list:
    """OpenCV-based heuristic demo when Roboflow client is unavailable."""
    h, w = img.shape[:2]
    detections = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Dark spots → holes
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
                "type": "hole", "confidence": round(conf, 3),
                "bbox": [x, y, x + bw, y + bh], "area_px": bw * bh,
                "severity": "high" if conf > 0.75 else "medium"
            })

    # Color blobs → strain
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, stain_mask = cv2.threshold(sat, 120, 255, cv2.THRESH_BINARY)
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel)
    contours2, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if 300 < area < (h * w * 0.20):
            x, y, bw, bh = cv2.boundingRect(cnt)
            conf = min(round(0.50 + area / (h * w * 2), 3), 0.90)
            detections.append({
                "type": "strain", "confidence": conf,
                "bbox": [x, y, x + bw, y + bh], "area_px": bw * bh,
                "severity": "medium" if conf > 0.6 else "low"
            })

    # Edge lines → tear
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
                "type": "tear", "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2], "area_px": (x2 - x1) * (y2 - y1),
                "severity": "high" if conf > 0.70 else "low"
            })

    return sorted(detections, key=lambda d: d["confidence"], reverse=True)[:6]


# ─── Main Analysis ────────────────────────────────────────────────────────────

def analyze_cloth(img: np.ndarray, image_bytes: bytes) -> dict:
    start = time.time()
    mode  = "simulation"

    if CLIENT is not None:
        try:
            detections = run_roboflow(image_bytes)
            mode = "roboflow"
        except Exception as e:
            logger.warning(f"Roboflow inference failed: {e}. Falling back to simulation.")
            detections = run_simulation(img)
    else:
        detections = run_simulation(img)

    elapsed = round(time.time() - start, 3)

    type_counts   = {}
    for d in detections:
        type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1

    summary_parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in type_counts.items()]
    summary = (f"{len(detections)} defect(s) found: {', '.join(summary_parts)}"
               if detections else "No defects detected — cloth appears clean ✅")

    if   len(detections) == 0: overall = "good"
    elif len(detections) <= 2: overall = "minor_damage"
    elif len(detections) <= 4: overall = "moderate_damage"
    else:                      overall = "severe_damage"

    return {
        "detections":          detections,
        "summary":             summary,
        "total_defects":       len(detections),
        "defect_types":        type_counts,
        "overall_condition":   overall,
        "processing_time_sec": elapsed,
        "mode":                mode
    }


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Cloth Defect Detection API is running 🧵", "version": "2.0.0"}


@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "client_loaded": CLIENT is not None,
        "backend":       "roboflow" if CLIENT else "simulation (inference_sdk not installed)",
        "workspace":     ROBOFLOW_WORKSPACE,
        "workflow":      ROBOFLOW_WORKFLOW,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload a cloth image and receive defect analysis results.
    Returns: detections list, annotated image (base64), summary stats.
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type: {file.content_type}. Use JPG/PNG.")

    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

    try:
        img = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result        = analyze_cloth(img, image_bytes)
    annotated_img = draw_detections(img, result["detections"])
    annotated_b64 = encode_image_base64(annotated_img)
    original_b64  = encode_image_base64(img)

    return JSONResponse({
        "status":          "success",
        "filename":        file.filename,
        "image_size":      {"width": img.shape[1], "height": img.shape[0]},
        "analysis":        result,
        "annotated_image": annotated_b64,
        "original_image":  original_b64
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)