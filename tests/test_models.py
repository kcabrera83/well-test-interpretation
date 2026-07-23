import pytest
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(MODELS_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


def test_analyzer_model_loads():
    from well_test_interpretation.models.pressure_analyzer import PressureAnalyzer
    path = os.path.join(MODELS_DIR, "pressure_analyzer.joblib")
    model = PressureAnalyzer.load(path)
    assert model is not None
    assert model.trained


def test_estimator_model_loads():
    from well_test_interpretation.models.reservoir_estimator import ReservoirEstimator
    path = os.path.join(MODELS_DIR, "reservoir_estimator.joblib")
    model = ReservoirEstimator.load(path)
    assert model is not None
    assert model.trained


def test_analyzer_prediction():
    import numpy as np
    from well_test_interpretation.models.pressure_analyzer import PressureAnalyzer
    model = PressureAnalyzer.load(os.path.join(MODELS_DIR, "pressure_analyzer.joblib"))
    X = np.random.rand(10, 9)
    preds = model.predict(X)
    assert preds is not None
    assert len(preds) == 10


def test_estimator_prediction():
    import numpy as np
    from well_test_interpretation.models.reservoir_estimator import ReservoirEstimator
    model = ReservoirEstimator.load(os.path.join(MODELS_DIR, "reservoir_estimator.joblib"))
    X = np.random.rand(10, 7)
    perm, skin = model.predict(X)
    assert perm is not None
    assert skin is not None
    assert len(perm) == 10
    assert len(skin) == 10
