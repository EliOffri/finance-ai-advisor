from fastapi import FastAPI
from app.api.prediction import router as prediction_router

app = FastAPI()
app.include_router(prediction_router, prefix="/api")
