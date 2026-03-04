# 🧵 FabricScan — Cloth Defect Detector

AI-powered web app that detects **holes, stains, tears, and fraying** in fabric images.

---

## 📁 Project Structure

```
cloth-defect-detector/
├── backend/
│   ├── main.py              ← FastAPI backend (main server)
│   ├── train_model.py       ← YOLOv8 training script
│   ├── requirements.txt     ← Python dependencies
│   └── Dockerfile           ← Docker config
├── frontend/
│   └── index.html           ← Complete frontend (single file)
├── models/
│   └── cloth_defect.pt      ← Your trained model goes here
└── README.md
```

---

## ⚡ QUICK START (5 Minutes)

### Step 1 — Clone / Download the Project

```bash
# If using git:
git clone <your-repo-url>
cd cloth-defect-detector

# Or just download and extract the ZIP
```

---

### Step 2 — Set Up Python Environment

```bash
# Make sure Python 3.9+ is installed
python --version

# Create a virtual environment (recommended)
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

---

### Step 3 — Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs:
- `fastapi` — web framework
- `uvicorn` — ASGI server
- `opencv-python-headless` — image processing
- `Pillow` — image utilities
- `numpy` — numerical operations

---

### Step 4 — Run the Backend Server

```bash
# Make sure you're in the backend/ folder
cd backend

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

✅ You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

> **Note:** Without a trained model file (`models/cloth_defect.pt`), the app runs in
> **Simulation Mode** using OpenCV-based heuristics. This still works for testing!

---

### Step 5 — Open the Frontend

```bash
# Option A: Just open the HTML file directly in your browser
# Double-click frontend/index.html
# OR right-click → Open With → Your Browser

# Option B: Serve with Python (avoids any CORS issues)
cd frontend
python -m http.server 3000
# Then open: http://localhost:3000
```

---

### Step 6 — Test It!

1. Open `http://localhost:3000` (or open `index.html` directly)
2. Drag & drop a cloth/fabric image (JPG, PNG, WEBP)
3. Click **Analyze Fabric**
4. See detected defects with bounding boxes + confidence scores

---

## 🤖 Adding a Real ML Model (YOLOv8)

### Step A — Install YOLOv8

```bash
pip install ultralytics torch torchvision
```

### Step B — Prepare Your Dataset

Organize labeled images like this:
```
datasets/cloth_defects/
├── images/
│   ├── train/    ← 700+ cloth images
│   └── val/      ← 150+ cloth images
└── labels/
    ├── train/    ← matching .txt files
    └── val/
```

**Labeling format** (YOLO `.txt` — one line per defect):
```
class_id  cx  cy  width  height
```
Where all values are normalized (0–1). Class IDs:
- `0` = hole
- `1` = stain
- `2` = tear
- `3` = fray

**Free labeling tools:**
- [Roboflow](https://roboflow.com) — best option, exports YOLO format
- [LabelImg](https://github.com/HumanSignal/labelImg)
- [CVAT](https://cvat.ai)

---

### Step C — Train the Model

```bash
cd backend
python train_model.py
```

Training runs for 100 epochs by default. The best model is automatically
copied to `models/cloth_defect.pt`.

**Metrics to aim for:**
- mAP50 > 0.80 = good
- mAP50-95 > 0.60 = good

---

### Step D — Test Inference

```bash
python train_model.py test path/to/your/test_image.jpg
```

---

## 🌐 API Reference

### `GET /health`
Check if server is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/path/to/models/cloth_defect.pt"
}
```

---

### `POST /analyze`
Upload a cloth image for defect analysis.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "status": "success",
  "filename": "cloth_sample.jpg",
  "image_size": { "width": 1024, "height": 768 },
  "analysis": {
    "detections": [
      {
        "type": "hole",
        "confidence": 0.94,
        "bbox": [120, 85, 210, 175],
        "area_px": 8100,
        "severity": "high"
      },
      {
        "type": "stain",
        "confidence": 0.87,
        "bbox": [300, 200, 390, 280],
        "area_px": 7200,
        "severity": "medium"
      }
    ],
    "summary": "2 defect(s) found: 1 hole, 1 stain",
    "total_defects": 2,
    "defect_types": { "hole": 1, "stain": 1 },
    "overall_condition": "minor_damage",
    "processing_time_sec": 0.12,
    "mode": "model"
  },
  "annotated_image": "<base64-encoded JPEG>",
  "original_image": "<base64-encoded JPEG>"
}
```

---

## 🐳 Docker Deployment

```bash
# Build
cd backend
docker build -t fabricscan-api .

# Run
docker run -p 8000:8000 -v $(pwd)/../models:/app/../models fabricscan-api
```

---

## ☁️ Cloud Deployment Options

| Platform | Steps |
|----------|-------|
| **Railway** | Push to GitHub → Connect to Railway → Auto-deploy |
| **Render** | Create Web Service → Set build command: `pip install -r requirements.txt` → Start: `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **AWS EC2** | SSH into instance → Clone repo → Run with Docker |
| **Google Cloud Run** | `gcloud run deploy` with Dockerfile |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot reach backend` | Make sure `uvicorn main:app --reload` is running |
| `CORS error` | Backend has CORS enabled for all origins by default |
| `Image decode error` | Use JPG, PNG, or WEBP format only |
| `Simulation mode only` | Install ultralytics + add model file to `models/` |
| `Port 8000 in use` | Use `--port 8001` and update `API` in `frontend/index.html` |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Backend | Python 3.11 + FastAPI |
| ML Model | YOLOv8 (Ultralytics) |
| Image Processing | OpenCV + Pillow |
| Deployment | Docker + Uvicorn |

---

## 🗺️ Roadmap

- [ ] Add batch image processing
- [ ] PDF inspection report export
- [ ] Defect severity heatmap
- [ ] React frontend upgrade
- [ ] WebSocket for real-time camera feed analysis