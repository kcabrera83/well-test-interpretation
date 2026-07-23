import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app, load_models

load_models()
client = app.test_client()


def _make_time_series(n=50):
    return {
        "time_hours": np.linspace(0.01, 100, n).tolist(),
        "pressure_psi": (3000 - 200 * np.log(np.linspace(0.01, 100, n))).tolist(),
        "flow_rate_bbl_d": [500.0] * n,
    }


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["analyzer"] is True
    assert data["models_loaded"]["estimator"] is True


def test_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "flow_regime_classifier" in data
    assert "reservoir_estimator" in data
    assert data["flow_regime_classifier"]["loaded"] is True
    assert data["reservoir_estimator"]["loaded"] is True


def test_api_docs():
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"


def test_analyze_valid():
    payload = _make_time_series(50)
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    assert "probabilities" in data
    assert data["n_samples"] == 50
    valid_regimes = {"Wellbore Storage", "Radial Flow", "Boundary Dominated"}
    for pred in data["predictions"]:
        assert pred in valid_regimes


def test_analyze_empty_data():
    response = client.post("/api/analyze", json={
        "time_hours": [],
        "pressure_psi": [],
        "flow_rate_bbl_d": [],
    })
    assert response.status_code == 400


def test_analyze_no_json():
    response = client.post("/api/analyze")
    assert response.status_code == 400


def test_estimate_valid():
    payload = _make_time_series(50)
    payload["wellbore_radius_ft"] = [0.328] * 50
    payload["reservoir_pressure_psi"] = [3500.0] * 50
    payload["drainage_radius_ft"] = [1500.0] * 50
    payload["formation_thickness_ft"] = [50.0] * 50
    response = client.post("/api/estimate", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "permeability_md" in data
    assert "skin_factor" in data
    assert data["n_samples"] == 50
    assert len(data["permeability_md"]) == 50
    assert len(data["skin_factor"]) == 50


def test_estimate_no_json():
    response = client.post("/api/estimate")
    assert response.status_code == 400
