from fastapi import APIRouter, HTTPException
from app.ml.model_manager import ModelManager

router = APIRouter()
manager = ModelManager()


@router.get("/predict/{ticker}")
def predict_default(ticker: str):
    try:
        result = manager.predict(ticker.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/predict/{model_name}/{ticker}")
def predict_with_model(model_name: str, ticker: str):
    try:
        result = manager.predict_with_model(model_name, ticker.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
