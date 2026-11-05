from pathlib import Path
import json
import joblib
import pandas as pd

MODEL_PATH = Path("models/fraud_model.joblib")
META_PATH = Path("models/metadata.json")

def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())

def score_transactions(data: pd.DataFrame):
    model, meta = load_artifacts()
    missing = set(meta["feature_columns"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    X = data[meta["feature_columns"]]
    probs = model.predict_proba(X)[:, 1]
    threshold = float(meta["threshold"])
    return pd.DataFrame({
        "fraud_probability": probs,
        "risk_decision": ["REVIEW" if p >= threshold else "LOW_RISK" for p in probs],
        "threshold": threshold
    })
