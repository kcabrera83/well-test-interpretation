"""FastAPI for well test interpretation."""

import os
import sys
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from well_test_interpretation.data_generator import generate_pressure_transient_data
from well_test_interpretation.utils.preprocessor import extract_features
from well_test_interpretation.models.pressure_analyzer import PressureAnalyzer
from well_test_interpretation.models.reservoir_estimator import ReservoirEstimator

app = FastAPI(
    title="Well Test Interpretation",
    description="Pressure transient analysis, flow regime classification, and reservoir estimation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
analyzer = None
estimator = None


@app.on_event("startup")
async def load_models():
    global analyzer, estimator
    analyzer_path = os.path.join(MODELS_DIR, "pressure_analyzer.joblib")
    estimator_path = os.path.join(MODELS_DIR, "reservoir_estimator.joblib")
    if os.path.exists(analyzer_path):
        analyzer = PressureAnalyzer.load(analyzer_path)
    if os.path.exists(estimator_path):
        estimator = ReservoirEstimator.load(estimator_path)


class AnalyzeRequest(BaseModel):
    time_hours: List[float]
    pressure_psi: List[float]
    flow_rate_bbl_d: List[float]


class AnalyzeResponse(BaseModel):
    predictions: List[str]
    probabilities: List[list]
    n_samples: int


class EstimateRequest(BaseModel):
    time_hours: List[float]
    pressure_psi: List[float]
    flow_rate_bbl_d: List[float]
    wellbore_radius_ft: List[float]
    reservoir_pressure_psi: List[float]
    drainage_radius_ft: List[float]
    formation_thickness_ft: List[float]


class EstimateResponse(BaseModel):
    permeability_md: List[float]
    skin_factor: List[float]
    n_samples: int


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "analyzer": analyzer is not None and analyzer.trained,
            "estimator": estimator is not None and estimator.trained,
        },
    }


@app.get("/api/models")
async def models_info():
    return {
        "flow_regime_classifier": {
            "type": "GradientBoostingClassifier",
            "loaded": analyzer is not None and analyzer.trained,
            "flow_regimes": PressureAnalyzer.FLOW_REGIMES if analyzer else {},
        },
        "reservoir_estimator": {
            "type": "RandomForestRegressor (permeability + skin)",
            "loaded": estimator is not None and estimator.trained,
        },
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    if analyzer is None or not analyzer.trained:
        raise HTTPException(status_code=503, detail="Flow regime model not loaded")
    n = len(request.time_hours)
    if n == 0:
        raise HTTPException(status_code=400, detail="Empty data arrays")
    try:
        df = generate_pressure_transient_data(n_samples=n, seed=0)
        df["time_hours"] = np.array(request.time_hours)
        df["pressure_psi"] = np.array(request.pressure_psi)
        df["flow_rate_bbl_d"] = np.array(request.flow_rate_bbl_d)
        df = extract_features(df)
        feature_cols = [
            "log_time", "pressure_psi", "dp_dlogt", "flow_rate_bbl_d",
            "normalized_pressure", "flow_normalized_dp", "pressure_squared",
            "wellbore_radius_ft", "formation_thickness_ft",
        ]
        X = df[feature_cols].values
        preds = analyzer.predict(X)
        probs = analyzer.predict_proba(X)
        regime_names = PressureAnalyzer.FLOW_REGIMES
        return AnalyzeResponse(
            predictions=[regime_names.get(int(p), f"Unknown({p})") for p in preds],
            probabilities=probs.tolist(),
            n_samples=n,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/estimate", response_model=EstimateResponse)
async def estimate(request: EstimateRequest):
    if estimator is None or not estimator.trained:
        raise HTTPException(status_code=503, detail="Reservoir estimator model not loaded")
    n = len(request.time_hours)
    if n == 0:
        raise HTTPException(status_code=400, detail="Empty data arrays")
    try:
        df = generate_pressure_transient_data(n_samples=n, seed=0)
        df["time_hours"] = np.array(request.time_hours)
        df["pressure_psi"] = np.array(request.pressure_psi)
        df["flow_rate_bbl_d"] = np.array(request.flow_rate_bbl_d)
        df["wellbore_radius_ft"] = np.array(request.wellbore_radius_ft)
        df["reservoir_pressure_psi"] = np.array(request.reservoir_pressure_psi)
        df["drainage_radius_ft"] = np.array(request.drainage_radius_ft)
        df["formation_thickness_ft"] = np.array(request.formation_thickness_ft)
        feature_cols = [
            "time_hours", "pressure_psi", "flow_rate_bbl_d",
            "wellbore_radius_ft", "reservoir_pressure_psi",
            "drainage_radius_ft", "formation_thickness_ft",
        ]
        X = df[feature_cols].values
        perm_pred, skin_pred = estimator.predict(X)
        return EstimateResponse(
            permeability_md=perm_pred.tolist(),
            skin_factor=skin_pred.tolist(),
            n_samples=n,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

