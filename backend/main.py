from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import os
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="TomatoGuard AI - Disease Detection API")

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Classes: {model.names}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# ── Disease Metadata ──────────────────────────────────────────────────────────
# Maps disease ID keys → metadata
DISEASE_META = {
    "bacterial_spot": {
        "id": "bacterial_spot",
        "name": "Bacterial Spot",
        "treatment": "Apply copper-based fungicides. Remove and destroy infected plant parts.",
        "severity": "High",
    },
    "early_blight": {
        "id": "early_blight",
        "name": "Early Blight",
        "treatment": "Apply fungicides containing chlorothalonil or copper. Improve air circulation.",
        "severity": "Medium",
    },
    "late_blight": {
        "id": "late_blight",
        "name": "Late Blight",
        "treatment": "Apply fungicides immediately. Remove infected plants. Highly contagious.",
        "severity": "Critical",
    },
    "leaf_mold": {
        "id": "leaf_mold",
        "name": "Leaf Mold",
        "treatment": "Reduce humidity, improve ventilation. Apply fungicides if necessary.",
        "severity": "Low",
    },
    "septoria_leaf_spot": {
        "id": "septoria_leaf_spot",
        "name": "Septoria Leaf Spot",
        "treatment": "Remove lower infected leaves. Apply organic fungicides.",
        "severity": "Medium",
    },
    "spider_mites": {
        "id": "spider_mites",
        "name": "Two-spotted Spider Mite",
        "treatment": "Use insecticidal soap or neem oil. Increase humidity.",
        "severity": "Medium",
    },
    "target_spot": {
        "id": "target_spot",
        "name": "Target Spot",
        "treatment": "Apply fungicides. Avoid overhead watering.",
        "severity": "Medium",
    },
    "yellow_leaf_curl_virus": {
        "id": "yellow_leaf_curl_virus",
        "name": "Yellow Leaf Curl Virus",
        "treatment": "Control whiteflies using insecticides or nets. Remove infected plants.",
        "severity": "High",
    },
    "mosaic_virus": {
        "id": "mosaic_virus",
        "name": "Mosaic Virus",
        "treatment": "No cure. Remove infected plants. Sanitize tools.",
        "severity": "High",
    },
    "black_spot": {
        "id": "black_spot",
        "name": "Black Spot",
        "treatment": "Remove infected leaves. Apply copper-based or sulfur fungicides. Improve air circulation and avoid leaf wetness.",
        "severity": "High",
    },
    "healthy": {
        "id": "healthy",
        "name": "Healthy",
        "treatment": "Keep up the good work! Maintain regular watering and monitoring.",
        "severity": "None",
    },
}

# ── Explicit class-name → disease ID mapping (based on actual model classes) ──
# Model classes: {0: 'Bacterial Spot', 1: 'Early_Blight', 2: 'Healthy',
#                 3: 'Late_blight', 4: 'Leaf Mold', 5: 'Target_Spot', 6: 'black spot'}
CLASS_NAME_MAP = {
    "bacterial spot":      "bacterial_spot",
    "bacterial_spot":      "bacterial_spot",
    "early_blight":        "early_blight",
    "early blight":        "early_blight",
    "healthy":             "healthy",
    "late_blight":         "late_blight",
    "late blight":         "late_blight",
    "leaf mold":           "leaf_mold",
    "leaf_mold":           "leaf_mold",
    "target_spot":         "target_spot",
    "target spot":         "target_spot",
    "black spot":          "black_spot",
    "black_spot":          "black_spot",
    # Fallback aliases for other possible class names
    "septoria_leaf_spot":  "septoria_leaf_spot",
    "septoria leaf spot":  "septoria_leaf_spot",
    "spider_mites":        "spider_mites",
    "two-spotted spider mite": "spider_mites",
    "yellow leaf curl virus": "yellow_leaf_curl_virus",
    "tomato yellow leaf curl virus": "yellow_leaf_curl_virus",
    "mosaic_virus":        "mosaic_virus",
    "tomato mosaic virus": "mosaic_virus",
}


def normalize_class_name(raw_name: str) -> str:
    """
    Normalize YOLOv8 class name to a disease ID key.
    Uses an explicit map first, then falls back to slug normalization.
    """
    lower = raw_name.lower().strip()
    # Try exact match in explicit map first
    if lower in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[lower]
    # Strip 'Tomato' prefix variants, then try again
    for prefix in ["tomato___", "tomato__", "tomato_", "tomato "]:
        if lower.startswith(prefix):
            stripped = lower[len(prefix):]
            if stripped in CLASS_NAME_MAP:
                return CLASS_NAME_MAP[stripped]
            lower = stripped
            break
    # Generic slug normalization
    slug = lower.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    return CLASS_NAME_MAP.get(slug, slug)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": list(model.names.values()) if model else [],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure best.pt is in backend/model/.",
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file (JPEG, PNG, etc.).",
        )

    # Read image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # Run inference
    try:
        results = model(image, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Parse detections
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            class_idx = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            raw_class = model.names[class_idx]
            key = normalize_class_name(raw_class)
            meta = DISEASE_META.get(key)
            if meta:
                detections.append({**meta, "confidence": round(confidence, 3)})
            else:
                # Unknown class — return raw name gracefully
                detections.append({
                    "id": key,
                    "disease_name": raw_class,
                    "confidence": round(confidence, 3),
                    "treatment": "Consult an agronomist for this disease type.",
                    "severity": "Unknown",
                })

    # If no detections, try classification fallback
    if not detections and hasattr(results[0], "probs") and results[0].probs is not None:
        probs = results[0].probs
        top_idx = int(probs.top1)
        top_conf = float(probs.top1conf.item())
        raw_class = model.names[top_idx]
        key = normalize_class_name(raw_class)
        meta = DISEASE_META.get(key)
        if meta:
            detections.append({**meta, "confidence": round(top_conf, 3)})

    # Deduplicate by disease_id (keep highest confidence per disease)
    seen = {}
    for d in detections:
        did = d.get("id", d.get("disease_id", ""))
        if did not in seen or d["confidence"] > seen[did]["confidence"]:
            seen[did] = d
    detections = sorted(seen.values(), key=lambda x: x["confidence"], reverse=True)

    if not detections:
        return {
            "success": False,
            "error": "No disease detected. Please upload a clear tomato leaf image.",
        }

    is_healthy = len(detections) == 1 and detections[0].get("id") == "healthy"

    # Rename id → disease_id for frontend compatibility
    diseases = []
    for d in detections:
        diseases.append({
            "disease_name": d.get("name", d.get("disease_name", "")),
            "disease_id": d.get("id", d.get("disease_id", "")),
            "confidence": d["confidence"],
            "treatment": d.get("treatment", ""),
            "severity": d.get("severity", ""),
        })

    return {
        "success": True,
        "isHealthy": is_healthy,
        "diseases": diseases,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
