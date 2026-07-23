import pytest
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_models_directory_exists():
    models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    assert os.path.exists(models_dir), f"Directory not found: {models_dir}"

def test_model_files_exist():
    models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    if not os.path.exists(models_dir):
        models_dir = os.path.join(PROJECT_ROOT, "models")
    model_files = [f for f in os.listdir(models_dir) if f.endswith((".pkl", ".joblib", ".h5", ".pt", ".json", ".onnx"))]
    assert len(model_files) > 0, f"No model files found in {models_dir}"