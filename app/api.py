# app/api.py
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

# model path relative to this file; normalise path for Windows
MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../models/house_rf.joblib"))
model = joblib.load(MODEL_PATH)

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def root():
    return {"status": "ok", "model": str(model.__class__.__name__)}

@app.post("/predict")
def predict(h: HouseFeatures):
    x = pd.DataFrame([h.dict()])
    pred = model.predict(x)[0]
    return {"prediction": float(pred)}
