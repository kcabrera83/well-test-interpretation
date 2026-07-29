
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib, os

app = FastAPI(title="Well Test Interpretation", description="Pressure transient analysis with flow regime identification")

class Measurement(BaseModel):
    features: dict = Field(..., example={"porosity": 0.15, "permeability": 100})

class Result(BaseModel):
    prediction: float
    confidence: float = 0.0

MODELS = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        MODELS[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))

@app.get("/")
def index():
    return {"service": "Well Test Interpretation", "models_available": list(MODELS.keys())}

@app.post("/evaluate/{model_name}")
def evaluate(model_name: str, meas: Measurement):
    m = MODELS.get(model_name)
    if not m:
        raise HTTPException(404, f"Model '{model_name}' not found")
    feats = m.get("feature_names", list(meas.features.keys()))
    X = np.array([meas.features.get(f, 0) for f in feats]).reshape(1, -1)
    if m.get("scaler"):
        X = m["scaler"].transform(X)
    pred = m["model"].predict(X)[0]
    conf = 0.0
    if hasattr(m["model"], "predict_proba"):
        conf = float(m["model"].predict_proba(X).max())
    return Result(prediction=float(pred), confidence=conf)
