"""API integration tests for Well Test Interpretation FastAPI app."""

import sys
from fastapi.testclient import TestClient

sys.path.insert(0, ".")
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    print("  [PASS] /api/health")
    return data


def test_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "flow_regime_classifier" in data
    assert "reservoir_estimator" in data
    assert data["flow_regime_classifier"]["loaded"] is True
    assert data["reservoir_estimator"]["loaded"] is True
    print("  [PASS] /api/models")
    return data


def test_analyze():
    payload = {
        "time_hours": [0.01, 0.1, 1.0, 10.0, 100.0],
        "pressure_psi": [5000.0, 4900.0, 4700.0, 4500.0, 4300.0],
        "flow_rate_bbl_d": [500.0, 500.0, 500.0, 500.0, 500.0],
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert len(data["predictions"]) == 5
    print(f"  [PASS] /api/analyze -> {set(data['predictions'])}")
    return data


def test_estimate():
    payload = {
        "time_hours": [0.01, 0.1, 1.0, 10.0, 100.0],
        "pressure_psi": [5000.0, 4900.0, 4700.0, 4500.0, 4300.0],
        "flow_rate_bbl_d": [500.0, 500.0, 500.0, 500.0, 500.0],
        "wellbore_radius_ft": [0.328, 0.328, 0.328, 0.328, 0.328],
        "reservoir_pressure_psi": [5500.0, 5500.0, 5500.0, 5500.0, 5500.0],
        "drainage_radius_ft": [1500.0, 1500.0, 1500.0, 1500.0, 1500.0],
        "formation_thickness_ft": [50.0, 50.0, 50.0, 50.0, 50.0],
    }
    r = client.post("/api/estimate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "permeability_md" in data
    assert "skin_factor" in data
    assert len(data["permeability_md"]) == 5
    avg_perm = sum(data["permeability_md"]) / len(data["permeability_md"])
    avg_skin = sum(data["skin_factor"]) / len(data["skin_factor"])
    print(f"  [PASS] /api/estimate -> perm={avg_perm:.1f} md, skin={avg_skin:.3f}")
    return data


def main():
    print("=" * 50)
    print("  WELL TEST INTERPRETATION - API TESTS")
    print("=" * 50)

    tests = [
        ("Health Check", test_health),
        ("Models Info", test_models),
        ("Flow Regime Analysis", test_analyze),
        ("Reservoir Estimation", test_estimate),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 50)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
