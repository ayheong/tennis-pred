from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict_names import predict_by_names

app = FastAPI()

class PredictionRequest(BaseModel):
    name_a: str
    name_b: str
    surface: Literal ["Hard", "Clay", "Grass", "Carpet"] = "Hard"
    best_of: Literal [3, 5] = 3

@app.post("/predict")
def predict(prediction_request: PredictionRequest):
    return predict_by_names(
        prediction_request.name_a,
        prediction_request.name_b,
        prediction_request.surface,
        prediction_request.best_of)

@app.get("/status")
def status():
    return {"status": "ok :)"}

