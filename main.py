import base64
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from image_processing.preprocess import preprocess_image
from model.mining_detector import (
    build_environmental_impact,
    classify_risk,
    detect_mining_regions,
)

app = FastAPI(title="Illegal Mining Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "Illegal Mining Detection API"}


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    data = await file.read()
    np_buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")

    preprocessed = preprocess_image(image)
    disturbed_regions, disturbed_ratio, boxed_image = detect_mining_regions(image, preprocessed)
    mining_detected, risk_level, confidence = classify_risk(disturbed_ratio, disturbed_regions)

    encoded_ok, encoded_buffer = cv2.imencode(".jpg", boxed_image)
    if not encoded_ok:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")

    processed_b64 = base64.b64encode(encoded_buffer.tobytes()).decode("utf-8")
    environmental_impact = build_environmental_impact(disturbed_ratio)

    return {
        "mining_detected": mining_detected,
        "risk_level": risk_level,
        "confidence": confidence,
        "disturbed_ratio": round(disturbed_ratio, 4),
        "environmental_impact": environmental_impact,
        "processed_image": processed_b64,
    }
