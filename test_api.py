"""API integration tests for well test interpretation application."""

import sys
import os
import json
import time
import requests

sys.path.insert(0, os.path.dirname(__file__))

BASE_URL = "http://127.0.0.1:5012"


def wait_for_server(timeout=10):
    """Wait for the Flask server to start."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/api/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            time.sleep(0.5)
    return False


def test_health():
    """Test /api/health endpoint."""
    r = requests.get(f"{BASE_URL}/api/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    data = r.json()
    assert data["status"] == "healthy"
    print("  [PASS] /api/health")
    return data


def test_models():
    """Test /api/models endpoint."""
    r = requests.get(f"{BASE_URL}/api/models")
    assert r.status_code == 200, f"Models check failed: {r.status_code}"
    data = r.json()
    assert "flow_regime_classifier" in data
    assert "reservoir_estimator" in data
    assert data["flow_regime_classifier"]["loaded"] is True
    assert data["reservoir_estimator"]["loaded"] is True
    print("  [PASS] /api/models")
    return data


def test_analyze():
    """Test /api/analyze endpoint."""
    payload = {
        "time_hours": [0.01, 0.1, 1.0, 10.0, 100.0],
        "pressure_psi": [5000.0, 4900.0, 4700.0, 4500.0, 4300.0],
        "flow_rate_bbl_d": [500.0, 500.0, 500.0, 500.0, 500.0],
    }
    r = requests.post(f"{BASE_URL}/api/analyze", json=payload)
    assert r.status_code == 200, f"Analyze failed: {r.status_code} {r.text}"
    data = r.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert len(data["predictions"]) == 5
    print(f"  [PASS] /api/analyze -> {set(data['predictions'])}")
    return data


def test_estimate():
    """Test /api/estimate endpoint."""
    payload = {
        "time_hours": [0.01, 0.1, 1.0, 10.0, 100.0],
        "pressure_psi": [5000.0, 4900.0, 4700.0, 4500.0, 4300.0],
        "flow_rate_bbl_d": [500.0, 500.0, 500.0, 500.0, 500.0],
        "wellbore_radius_ft": [0.328, 0.328, 0.328, 0.328, 0.328],
        "reservoir_pressure_psi": [5500.0, 5500.0, 5500.0, 5500.0, 5500.0],
        "drainage_radius_ft": [1500.0, 1500.0, 1500.0, 1500.0, 1500.0],
        "formation_thickness_ft": [50.0, 50.0, 50.0, 50.0, 50.0],
    }
    r = requests.post(f"{BASE_URL}/api/estimate", json=payload)
    assert r.status_code == 200, f"Estimate failed: {r.status_code} {r.text}"
    data = r.json()
    assert "permeability_md" in data
    assert "skin_factor" in data
    assert len(data["permeability_md"]) == 5
    avg_perm = sum(data["permeability_md"]) / len(data["permeability_md"])
    avg_skin = sum(data["skin_factor"]) / len(data["skin_factor"])
    print(f"  [PASS] /api/estimate -> perm={avg_perm:.1f} md, skin={avg_skin:.3f}")
    return data


def test_index():
    """Test / (index) endpoint."""
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Index failed: {r.status_code}"
    assert "text/html" in r.headers.get("Content-Type", "")
    print("  [PASS] / (index HTML)")
    return True


def main():
    """Run all API tests."""
    print("=" * 50)
    print("  WELL TEST INTERPRETATION - API TESTS")
    print("=" * 50)

    print("\nWaiting for server...")
    if not wait_for_server():
        print("  [FAIL] Server did not start within timeout")
        sys.exit(1)
    print("  Server is up.\n")

    tests = [
        ("Health Check", test_health),
        ("Models Info", test_models),
        ("Flow Regime Analysis", test_analyze),
        ("Reservoir Estimation", test_estimate),
        ("Index Page", test_index),
    ]

    passed = 0
    failed = 0
    results = {}

    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = {"status": "PASS", "data": result}
            passed += 1
        except AssertionError as e:
            results[name] = {"status": "FAIL", "error": str(e)}
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            results[name] = {"status": "ERROR", "error": str(e)}
            print(f"  [ERROR] {name}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
