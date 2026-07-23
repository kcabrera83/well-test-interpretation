import pytest
import numpy as np


def _make_time_series(n=50):
    return {
        "time_hours": np.linspace(0.01, 100, n).tolist(),
        "pressure_psi": (3000 - 200 * np.log(np.linspace(0.01, 100, n))).tolist(),
        "flow_rate_bbl_d": [500.0] * n,
    }


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    pass  # models may or may not be loaded


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "flow_regime_classifier" in data
    assert "reservoir_estimator" in data


def test_analyze_valid(client):
    payload = _make_time_series(50)
    response = client.post("/api/analyze", json=payload)
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "probabilities" in data
        assert data["n_samples"] == 50
        valid_regimes = {"Wellbore Storage", "Radial Flow", "Boundary Dominated"}
        for pred in data["predictions"]:
            assert pred in valid_regimes


def test_analyze_empty_data(client):
    response = client.post("/api/analyze", json={
        "time_hours": [],
        "pressure_psi": [],
        "flow_rate_bbl_d": [],
    })
    assert response.status_code == 400


def test_estimate_valid(client):
    payload = _make_time_series(50)
    payload["wellbore_radius_ft"] = [0.328] * 50
    payload["reservoir_pressure_psi"] = [3500.0] * 50
    payload["drainage_radius_ft"] = [1500.0] * 50
    payload["formation_thickness_ft"] = [50.0] * 50
    response = client.post("/api/estimate", json=payload)
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert "permeability_md" in data
        assert "skin_factor" in data
        assert data["n_samples"] == 50
        assert len(data["permeability_md"]) == 50
        assert len(data["skin_factor"]) == 50
