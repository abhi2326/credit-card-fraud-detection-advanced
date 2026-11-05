from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
from src.predict import score_transactions

app = FastAPI(title="Fraud Detection API", version="1.0.0",
              description="Portfolio demonstration fraud-risk scoring endpoint.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Accept either {"features": {...}} or a flat feature dictionary.
    features = payload.get("features", payload)
    try:
        result = score_transactions(pd.DataFrame([features])).iloc[0].to_dict()
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
