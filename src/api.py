"""
DetectX FastAPI Backend Service.

Provides:
- GET /health: Model health status and metadata
- POST /predict: Inference using the fitted Random Forest scikit-learn pipeline
- GET /transactions: Real transaction screening session history
- GET /stats: Aggregated session metrics and daily trend data
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectx.api")

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
DATASET_REPORT_PATH = ROOT / "reports" / "dataset_analysis.json"

# Load model pipeline
if not MODEL_PATH.exists():
    logger.error("Model pipeline file not found at %s", MODEL_PATH)
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Successfully loaded ML pipeline from %s", MODEL_PATH)
except Exception as exc:
    logger.exception("Failed to load model pipeline: %s", exc)
    model = None

# Load model decision threshold from metadata if available
THRESHOLD = 0.5
if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            THRESHOLD = float(meta.get("metrics", {}).get("threshold", 0.5))
            logger.info("Loaded decision threshold from metadata: %s", THRESHOLD)
    except Exception as exc:
        logger.warning("Could not read threshold from metadata, default to 0.5: %s", exc)

app = FastAPI(
    title="DetectX API",
    description="Machine-learning financial fraud detection API",
    version="1.0.0",
)

# Allowed frontend origins (local dev + optional env-provided domains)
ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

env_origins = os.environ.get("ADDITIONAL_CORS_ORIGINS", "")
if env_origins:
    for origin in env_origins.split(","):
        clean_origin = origin.strip().rstrip("/")
        if clean_origin and clean_origin not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(clean_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$|^https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

ALLOWED_TYPES = {
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "DEBIT",
}

# Session-based transaction store
history: List[dict] = []
history_lock = Lock()


class Transaction(BaseModel):
    step: int = Field(ge=1, description="Hour of simulation (>= 1)")
    type: str = Field(description="Transaction type: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT")
    amount: float = Field(ge=0.0, allow_inf_nan=False, description="Transaction amount")
    oldbalanceOrg: float = Field(ge=0.0, allow_inf_nan=False, description="Sender balance before")
    newbalanceOrig: float = Field(ge=0.0, allow_inf_nan=False, description="Sender balance after")
    oldbalanceDest: float = Field(ge=0.0, allow_inf_nan=False, description="Recipient balance before")
    newbalanceDest: float = Field(ge=0.0, allow_inf_nan=False, description="Recipient balance after")


@app.get("/")
def root():
    return {
        "name": "DetectX API",
        "status": "online",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline is not loaded or unavailable",
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "model": "RandomForest Pipeline",
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is offline")

    transaction_type = transaction.type.strip().upper()
    if transaction_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported transaction type '{transaction.type}'. Allowed types: {sorted(ALLOWED_TYPES)}",
        )

    try:
        row = {
            "step": int(transaction.step),
            "type": transaction_type,
            "amount": float(transaction.amount),
            "oldbalanceOrg": float(transaction.oldbalanceOrg),
            "newbalanceOrig": float(transaction.newbalanceOrig),
            "oldbalanceDest": float(transaction.oldbalanceDest),
            "newbalanceDest": float(transaction.newbalanceDest),
        }

        # Construct dataframe matching pipeline schema
        frame = pd.DataFrame([[row[col] for col in FEATURES]], columns=FEATURES)

        # Run pipeline inference
        raw_prediction = int(model.predict(frame)[0])

        fraud_probability: Optional[float] = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)[0]
            classes = list(model.classes_)
            if 1 in classes:
                fraud_probability = float(probabilities[classes.index(1)])

        # Apply threshold
        if fraud_probability is not None:
            is_fraud = bool(fraud_probability >= THRESHOLD)
        else:
            is_fraud = bool(raw_prediction == 1)

        prediction_int = 1 if is_fraud else 0

        # Assess risk level
        if fraud_probability is None:
            risk_level = "High" if is_fraud else "Low"
        elif fraud_probability >= 0.7:
            risk_level = "High"
        elif fraud_probability >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        with history_lock:
            record_id = f"TX-{len(history) + 1:05d}"
            result = {
                "id": record_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **row,
                "prediction": prediction_int,
                "is_fraud": is_fraud,
                "probability": fraud_probability,
                "fraud_probability": fraud_probability,
                "label": "Fraudulent" if is_fraud else "Legitimate",
                "risk_level": risk_level,
            }
            history.append(result.copy())

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Inference processing failed. Please check feature values.",
        ) from exc


@app.get("/transactions")
def get_transactions(limit: int = 100):
    if not 1 <= limit <= 1000:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 1000",
        )

    with history_lock:
        items = list(reversed(history[-limit:]))

    return items


@app.get("/stats")
def get_stats():
    with history_lock:
        items = list(history)

    total = len(items)
    fraud_count = sum(1 for item in items if item["prediction"] == 1)
    legitimate_count = total - fraud_count
    fraud_rate = round((fraud_count / total) * 100, 2) if total > 0 else 0.0

    # Build daily distribution for UI activity charts
    daily_map = {}
    for item in items:
        try:
            dt = datetime.fromisoformat(item["timestamp"])
            day_str = dt.strftime("%Y-%m-%d")
        except Exception:
            day_str = "Today"

        if day_str not in daily_map:
            daily_map[day_str] = {"date": day_str, "total": 0, "fraud": 0}
        daily_map[day_str]["total"] += 1
        if item["prediction"] == 1:
            daily_map[day_str]["fraud"] += 1

    daily_list = list(daily_map.values())

    return {
        "total": total,
        "fraud": fraud_count,
        "legitimate": legitimate_count,
        "fraud_rate": fraud_rate,
        "daily": daily_list,
        # Legacy key names for backward compatibility
        "total_transactions": total,
        "fraudulent_transactions": fraud_count,
        "legitimate_transactions": legitimate_count,
    }


@app.get("/dataset-stats")
def get_dataset_stats():
    if not DATASET_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset analysis report not found at reports/dataset_analysis.json",
        )
    try:
        with open(DATASET_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        logger.exception("Failed to load dataset analysis report: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to read dataset analysis report",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info("Starting DetectX API on %s:%d", host, port)
    uvicorn.run("src.api:app", host=host, port=port)

