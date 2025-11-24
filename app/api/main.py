from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from app.utils.realtime import compute_latest_features, FEATURE_ORDER


app = FastAPI()

MODEL_DIR = Path("app/models")

lr_model = joblib.load(MODEL_DIR / "logistic_regression.pkl")
EXPECTED_FEATURE_COUNT = lr_model.n_features_in_  # <--- IMPORTANT


class Input(BaseModel):
    feature_values: list[float]


@app.post("/predict")
def predict(input: Input):
    # Validate feature count
    if len(input.feature_values) != EXPECTED_FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid number of features: expected {EXPECTED_FEATURE_COUNT}, got {len(input.feature_values)}"
        )

    # Convert to DataFrame
    X = pd.DataFrame([input.feature_values])

    # Compute probability
    prob_up = float(lr_model.predict_proba(X)[0][1])
    direction = "UP" if prob_up >= 0.5 else "DOWN"

    if prob_up > 0.65:
        signal = "BUY"
    elif prob_up < 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "probability_up": prob_up,
        "direction": direction,
        "signal": signal,
    }


@app.get("/predict/tomorrow")
def predict_tomorrow(ticker: str):
    features = compute_latest_features(ticker)

    if features is None:
        raise HTTPException(
            status_code=422,
            detail="Not enough data to compute features for this ticker."
        )

    X = pd.DataFrame([features])

    prob_up = float(lr_model.predict_proba(X)[0][1])
    direction = "UP" if prob_up >= 0.5 else "DOWN"

    if prob_up > 0.65:
        signal = "BUY"
    elif prob_up < 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ticker": ticker,
        "probability_up": prob_up,
        "direction": direction,
        "signal": signal,
        "features_used": FEATURE_ORDER
    }
