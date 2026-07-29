import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Well Test Interpretation", layout="wide")
st.title("Well Test Interpretation")
st.markdown("Identify flow regimes and estimate reservoir parameters.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'regime': joblib.load(d / 'flow_regime_classifier.pkl'), 'perm': joblib.load(d / 'permeability_estimator.pkl'), 'skin': joblib.load(d / 'skin_estimator.pkl')}

st.sidebar.header("Input Parameters")
time = st.sidebar.slider('Time', 0, 1000, 500)
pressure = st.sidebar.slider('Pressure', 1000, 10000, 5500)
derivative = st.sidebar.slider('Derivative', 0, 1000, 500)
rate = st.sidebar.slider('Rate', 100, 10000, 5050)

if st.sidebar.button("Run"):
    try:
        x = np.array([[time, pressure, derivative, rate]])
        cols = st.columns(3)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))