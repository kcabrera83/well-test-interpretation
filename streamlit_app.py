import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Well Test Interpretation", layout="wide")
st.title("Well Test Interpretation")
st.markdown("Interpret pressure transient analysis data.")

try:
    import joblib
    import numpy as np
    d = Path(__file__).parent / "outputs" / "models"
    models = {}
    for k, v in [("regime", "flow_regime_classifier.pkl"), ("permeability", "permeability_estimator.pkl"), ("skin", "skin_estimator.pkl")]:
        models[k] = joblib.load(d / v)
    st.success("Models loaded successfully")
except Exception as e:
    st.error(f"Model loading error: {type(e).__name__}: {e}")
    st.stop()

st.sidebar.header("Input Parameters")
time_hr = st.sidebar.slider("Time (hr)", 0, 1000, 500)
pressure_psi = st.sidebar.slider("Pressure (psi)", 1000, 10000, 5500)
derivative_psi = st.sidebar.slider("Derivative (psi)", 0, 1000, 500)
rate_bbl_day = st.sidebar.slider("Rate (bbl/d)", 100, 10000, 5050)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[time_hr, pressure_psi, derivative_psi, rate_bbl_day]])
        results = {}
        for name, key in [("Regime", "regime"), ("Permeability (md)", "permeability"), ("Skin", "skin")]:
            m = models[key]
            X = m["scaler"].transform(features)
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                results[name] = m["label_encoder"].inverse_transform(pred)[0]
            else:
                results[name] = f"{pred[0]:.2f}"
        cols = st.columns(3)
        for i, (k, v) in enumerate(results.items()):
            cols[i].metric(k, v)
    except Exception as e:
        st.error(f"Prediction error: {e}")
