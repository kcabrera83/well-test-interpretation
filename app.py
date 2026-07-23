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
    version="2.0.0",
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
    try:
        analyzer_path = os.path.join(MODELS_DIR, "pressure_analyzer.joblib")
        estimator_path = os.path.join(MODELS_DIR, "reservoir_estimator.joblib")
        if os.path.exists(analyzer_path):
            analyzer = PressureAnalyzer.load(analyzer_path)
        if os.path.exists(estimator_path):
            estimator = ReservoirEstimator.load(estimator_path)
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")


class AnalyzeRequest(BaseModel):
    time_hours: List[float]
    pressure_psi: List[float]
    flow_rate_bbl_d: List[float]
    wellbore_radius_ft: float = 0.328
    reservoir_pressure_psi: float = 3000.0
    formation_thickness_ft: float = 50.0


class AnalyzeResponse(BaseModel):
    predictions: List[str]
    probabilities: List[list]
    n_samples: int


class EstimateRequest(BaseModel):
    time_hours: List[float]
    pressure_psi: List[float]
    flow_rate_bbl_d: List[float]
    wellbore_radius_ft: float
    reservoir_pressure_psi: float
    drainage_radius_ft: float
    formation_thickness_ft: float


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
        "flow_regime_analyzer": {
            "type": "lmfit curve fitting + Bayesian classification",
            "loaded": analyzer is not None and analyzer.trained,
            "flow_regimes": PressureAnalyzer.FLOW_REGIMES if analyzer else {},
        },
        "reservoir_estimator": {
            "type": "lmfit parametric curve fitting (permeability + skin)",
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
        import pandas as pd
        df = pd.DataFrame({
            "time_hours": np.array(request.time_hours),
            "pressure_psi": np.array(request.pressure_psi),
            "flow_rate_bbl_d": np.array(request.flow_rate_bbl_d),
            "wellbore_radius_ft": request.wellbore_radius_ft,
            "reservoir_pressure_psi": request.reservoir_pressure_psi,
            "formation_thickness_ft": request.formation_thickness_ft,
            "permeability_md": np.ones(n),
            "skin_factor": np.zeros(n),
            "drainage_radius_ft": np.ones(n) * 1000.0,
        })
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
        import pandas as pd
        df = pd.DataFrame({
            "time_hours": np.array(request.time_hours),
            "pressure_psi": np.array(request.pressure_psi),
            "flow_rate_bbl_d": np.array(request.flow_rate_bbl_d),
            "wellbore_radius_ft": request.wellbore_radius_ft,
            "reservoir_pressure_psi": request.reservoir_pressure_psi,
            "drainage_radius_ft": request.drainage_radius_ft,
            "formation_thickness_ft": request.formation_thickness_ft,
            "permeability_md": np.ones(n),
            "skin_factor": np.zeros(n),
        })
        df = extract_features(df)
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
