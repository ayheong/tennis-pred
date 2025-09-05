# app.py
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.predict_names import predict_by_names

app = FastAPI()

class PredictionRequest(BaseModel):
    name_a: str
    name_b: str
    surface: Literal["Hard", "Clay", "Grass", "Carpet"] = "Hard"
    best_of: Literal[3, 5] = 3
    tourney_level: str = "A"          # optional; passes through
    as_of: Optional[str] = None       # "YYYY-MM-DD" or None

@app.post("/predict")
def predict(prediction_request: PredictionRequest):
    try:
        result = predict_by_names(
            name_a=prediction_request.name_a,
            name_b=prediction_request.name_b,
            surface=prediction_request.surface,
            best_of=prediction_request.best_of,
            tourney_level=prediction_request.tourney_level,
            as_of=prediction_request.as_of,
        )
        return result
    except ValueError as e:  # e.g., player name not found
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:  # missing models or data files
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    return {"status": "ok :)"}
