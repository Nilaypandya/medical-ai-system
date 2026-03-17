"""
Medical AI System - Main FastAPI Application
=============================================
Entry point for the REST API. Handles routing, middleware,
CORS configuration, and monitoring data collection.
"""

import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.inference import run_inference
from app.model_loader import load_all_models


# ---------------------------------------------------------------------------
# Monitoring Store
# ---------------------------------------------------------------------------

monitoring_store: dict[str, Any] = {
    "total_predictions": 0,
    "inference_times": deque(maxlen=1000),
    "prediction_counts": {},
    "confidence_scores": deque(maxlen=1000),
    "recent_predictions": deque(maxlen=50),
}

SUPPORTED_DISEASES = ["brain_tumor", "pneumonia", "skin_cancer"]


# ---------------------------------------------------------------------------
# Medical Risk Levels
# ---------------------------------------------------------------------------

RISK_LEVELS = {

    # Brain Tumor
    "glioma": "high",
    "meningioma": "medium",
    "pituitary": "medium",
    "no_tumor": "safe",

    # Pneumonia
    "pneumonia": "high",
    "normal": "safe",

    # Skin Cancer
    "melanoma": "high",
    "basal_cell_carcinoma": "high",
    "actinic_keratosis": "medium",
    "benign_keratosis": "safe",
    "melanocytic_nevi": "safe",
    "dermatofibroma": "safe",
    "vascular_lesion": "safe",
}


# ---------------------------------------------------------------------------
# Generate Medical Warning
# ---------------------------------------------------------------------------

def generate_warning(label: str, confidence: float):

    risk = RISK_LEVELS.get(label, "unknown")

    if risk == "high":
        return {
            "level": "HIGH",
            "message": "⚠ High risk detected. Please consult a medical specialist immediately."
        }

    if risk == "medium":
        return {
            "level": "MEDIUM",
            "message": "⚠ Moderate risk detected. Medical evaluation recommended."
        }

    if risk == "safe":
        return {
            "level": "LOW",
            "message": "✓ Low risk. No serious abnormality detected."
        }

    return {
        "level": "UNKNOWN",
        "message": "Unable to determine medical risk."
    }


# ---------------------------------------------------------------------------
# Startup Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("🚀 Loading medical AI models …")

    load_all_models()

    print("✅ Models ready")

    yield

    print("👋 Shutting down")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Image AI System",
    description="Multi-disease detection from medical images",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# ---------------------------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------------------------

@app.post("/predict", tags=["Inference"])
async def predict(

    file: UploadFile = File(...),
    disease_type: str = Form(...)

):

    # Validate disease type

    if disease_type not in SUPPORTED_DISEASES:

        raise HTTPException(
            status_code=400,
            detail=f"disease_type must be one of {SUPPORTED_DISEASES}"
        )

    # Validate file type

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):

        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported"
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:

        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Run inference

    t0 = time.perf_counter()

    result = await run_inference(image_bytes, disease_type)

    elapsed = round(time.perf_counter() - t0, 4)

    label = result["prediction"]

    confidence = result["confidence"]

    # Generate medical warning

    warning = generate_warning(label, confidence)

    # Update monitoring

    monitoring_store["total_predictions"] += 1

    monitoring_store["inference_times"].append(elapsed)

    monitoring_store["confidence_scores"].append(confidence)

    monitoring_store["prediction_counts"][label] = (
        monitoring_store["prediction_counts"].get(label, 0) + 1
    )

    monitoring_store["recent_predictions"].appendleft(
        {
            "id": str(uuid.uuid4())[:8],
            "disease_type": disease_type,
            "prediction": label,
            "confidence": confidence,
            "risk_level": warning["level"],
            "inference_time": elapsed,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Response

    return JSONResponse(
        {
            "prediction": label,
            "confidence": confidence,
            "risk_level": warning["level"],
            "medical_warning": warning["message"],
            "heatmap": result["heatmap"],
            "all_probabilities": result["all_probabilities"],
            "inference_time_seconds": elapsed,
            "disease_type": disease_type,
        }
    )


# ---------------------------------------------------------------------------
# Monitoring Endpoint
# ---------------------------------------------------------------------------

@app.get("/monitoring", tags=["Monitoring"])
async def monitoring():

    times = list(monitoring_store["inference_times"])

    scores = list(monitoring_store["confidence_scores"])

    avg_time = round(sum(times) / len(times), 4) if times else 0

    avg_conf = round(sum(scores) / len(scores), 4) if scores else 0

    min_conf = round(min(scores), 4) if scores else 0

    max_conf = round(max(scores), 4) if scores else 0

    return {

        "total_predictions": monitoring_store["total_predictions"],

        "average_inference_time_seconds": avg_time,

        "average_confidence": avg_conf,

        "min_confidence": min_conf,

        "max_confidence": max_conf,

        "prediction_distribution": monitoring_store["prediction_counts"],

        "recent_predictions": list(monitoring_store["recent_predictions"])[:10],

    }


# ---------------------------------------------------------------------------
# Disease Metadata
# ---------------------------------------------------------------------------

@app.get("/diseases", tags=["Metadata"])
async def diseases():

    return {

        "brain_tumor": {

            "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],

            "modality": "MRI",

        },

        "pneumonia": {

            "classes": ["normal", "pneumonia"],

            "modality": "Chest X-Ray",

        },

        "skin_cancer": {

            "classes": [

                "actinic_keratosis",

                "basal_cell_carcinoma",

                "benign_keratosis",

                "dermatofibroma",

                "melanocytic_nevi",

                "melanoma",

                "vascular_lesion",

            ],

            "modality": "Dermoscopy",

        },

    }


# ---------------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)