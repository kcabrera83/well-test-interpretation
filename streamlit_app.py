import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Well Test Interpretation", layout="wide")
st.title("Well Test Interpretation")
st.markdown("Interpret pressure transient analysis data to identify flow regimes and estimate reservoir parameters.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("regime", "flow_regime_classifier.pkl"), ("permeability", "permeability_estimator.pkl"), ("skin", "skin_estimator.pkl")]}

try:
    models = load_models()
    st.success('Models loaded successfully')
except Exception as e:
    st.error(f'Failed to load models: {e}')
    st.stop()

st.sidebar.header("Input Parameters")
time_hr = st.sidebar.slider("Time Hr", 0, 1000, 500)
pressure_psi = st.sidebar.slider("Pressure Psi", 1000, 10000, 5500)
derivative_psi = st.sidebar.slider("Derivative Psi", 0, 1000, 500)
rate_bbl_day = st.sidebar.slider("Rate Bbl Day", 100, 10000, 5050)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[time_hr, pressure_psi, derivative_psi, rate_bbl_day]])
        m = models["regime"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Regime", result if isinstance(result, str) else f"{result:.4f}")
        m = models["permeability"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Permeability", result if isinstance(result, str) else f"{result:.4f}")
        m = models["skin"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Skin", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")


