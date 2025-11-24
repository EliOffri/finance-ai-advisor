# Finance AI Advisor

A full end‑to‑end machine‑learning system that predicts next‑day stock returns using high‑quality technical indicators, PostgreSQL storage, and FastAPI inference.

## 🚀 Features
- Automatic OHLCV data ingestion
- 40+ engineered technical indicators (RSI, MACD, MFI, OBV, ATR, Bollinger Bands, etc.)
- JSONB feature storage in PostgreSQL
- Time‑series aligned training set
- ML models (Linear Regression & Random Forest Regressor)
- Preprocessing pipeline with scaling
- FastAPI prediction endpoint
- Modular, production‑ready project structure

## 📁 Project Structure
```
app/
  ├── api/
  ├── ml/
  │     └── features.py
  ├── models/
  ├── utils/
scripts/
  ├── load_features_to_db.py
  ├── train_model.py
data/
```

## 🧠 ML Workflow
1. Load OHLCV data → compute indicators  
2. Store technical features as JSONB in PostgreSQL  
3. Build training set aligned per ticker  
4. Train ML regressors  
5. Serialize models to `app/models/`  
6. Serve predictions via FastAPI  

## ▶️ Running
### Load indicators
```
uv run python -m scripts.load_features_to_db
```

### Train models
```
uv run python -m scripts.train_model
```

### Start API
```
uv run uvicorn app.api.main:app --reload
```

## 📝 Prediction API
POST `/predict`

```json
{
  "feature_values": [1.02, -0.003, 54.1, ...]
}
```

## 📦 Models
Models are saved under:
```
app/models/
  ├── linear_regression.pkl
  └── random_forest.pkl
```

## 📄 License
MIT
