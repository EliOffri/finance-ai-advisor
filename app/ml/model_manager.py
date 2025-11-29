import joblib
from pathlib import Path
import pandas as pd
from sqlalchemy import text

from app.utils.db import get_conn
from app.ml.features import compute_features


class ModelManager:
    def __init__(self):
        self.model_dir = Path("app/models")
        self.models = {}
        self.default_model_name = "extra_trees"

        self.load_models()

    # Load all .pkl models once
    def load_models(self):
        print("Loading ML models...")
        for p in self.model_dir.glob("*.pkl"):
            name = p.stem
            self.models[name] = joblib.load(p)
            print(f"Loaded model: {name}")

    # Fetch recent OHLCV from DB
    def fetch_latest_data(self, ticker, days=60):
        query = text("""
            SELECT date, open, high, low, close, volume
            FROM ohlcv
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT :limit;
        """)

        conn = get_conn()
        df = pd.read_sql(query, conn, params={"ticker": ticker, "limit": days})
        if df.empty:
            raise ValueError(f"No OHLCV data found for ticker {ticker}")

        df = df.sort_values("date")  # ensure ascending order
        return df

    # Compute features and return last row only
    def prepare_features(self, df):
        df = compute_features(df)
        latest = df.iloc[-1:]
        X = latest.drop(columns=["date"], errors="ignore")
        return X

    # Predict with specific model
    def predict_with_model(self, model_name, ticker):
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.models[model_name]

        df = self.fetch_latest_data(ticker)
        X = self.prepare_features(df)

        pred = float(model.predict(X)[0])
        signal = "LONG" if pred > 0 else "SHORT"

        return {"model": model_name, "prediction": pred, "signal": signal}

    # Predict using default (best) model
    def predict(self, ticker):
        return self.predict_with_model(self.default_model_name, ticker)
