"""Flask web application for well test interpretation."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from well_test_interpretation.data_generator import generate_pressure_transient_data
from well_test_interpretation.utils.preprocessor import extract_features
from well_test_interpretation.models.pressure_analyzer import PressureAnalyzer
from well_test_interpretation.models.reservoir_estimator import ReservoirEstimator

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
analyzer = None
estimator = None


def load_models():
    """Load trained models from disk."""
    global analyzer, estimator
    analyzer_path = os.path.join(MODELS_DIR, "pressure_analyzer.joblib")
    estimator_path = os.path.join(MODELS_DIR, "reservoir_estimator.joblib")
    if os.path.exists(analyzer_path):
        analyzer = PressureAnalyzer.load(analyzer_path)
    if os.path.exists(estimator_path):
        estimator = ReservoirEstimator.load(estimator_path)


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze pressure data to identify flow regimes.

    Expects JSON with:
        time_hours: list[float]
        pressure_psi: list[float]
        flow_rate_bbl_d: list[float]

    Returns:
        JSON with flow regime predictions and probabilities.
    """
    if analyzer is None or not analyzer.trained:
        return jsonify({"error": "Flow regime model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        n = len(data.get("time_hours", []))
        if n == 0:
            return jsonify({"error": "Empty data arrays"}), 400

        df = generate_pressure_transient_data(n_samples=n, seed=0)
        df["time_hours"] = np.array(data["time_hours"])
        df["pressure_psi"] = np.array(data["pressure_psi"])
        df["flow_rate_bbl_d"] = np.array(data["flow_rate_bbl_d"])

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

        return jsonify({
            "predictions": [regime_names.get(int(p), f"Unknown({p})") for p in preds],
            "probabilities": probs.tolist(),
            "n_samples": n,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/estimate", methods=["POST"])
def estimate():
    """Estimate permeability and skin factor from pressure data.

    Expects JSON with:
        time_hours: list[float]
        pressure_psi: list[float]
        flow_rate_bbl_d: list[float]
        wellbore_radius_ft: list[float]
        reservoir_pressure_psi: list[float]
        drainage_radius_ft: list[float]
        formation_thickness_ft: list[float]

    Returns:
        JSON with permeability_md and skin_factor estimates.
    """
    if estimator is None or not estimator.trained:
        return jsonify({"error": "Reservoir estimator model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        n = len(data.get("time_hours", []))
        if n == 0:
            return jsonify({"error": "Empty data arrays"}), 400

        df = generate_pressure_transient_data(n_samples=n, seed=0)
        for key in ["time_hours", "pressure_psi", "flow_rate_bbl_d",
                     "wellbore_radius_ft", "reservoir_pressure_psi",
                     "drainage_radius_ft", "formation_thickness_ft"]:
            if key in data:
                df[key] = np.array(data[key])

        feature_cols = [
            "time_hours", "pressure_psi", "flow_rate_bbl_d",
            "wellbore_radius_ft", "reservoir_pressure_psi",
            "drainage_radius_ft", "formation_thickness_ft",
        ]
        X = df[feature_cols].values
        perm_pred, skin_pred = estimator.predict(X)

        return jsonify({
            "permeability_md": perm_pred.tolist(),
            "skin_factor": skin_pred.tolist(),
            "n_samples": n,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def models():
    """Return information about loaded models."""
    model_info = {
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
    return jsonify(model_info)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "analyzer": analyzer is not None and analyzer.trained,
            "estimator": estimator is not None and estimator.trained,
        },
    })


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5012, debug=False)
