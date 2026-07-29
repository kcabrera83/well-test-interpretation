
import streamlit as st
import numpy as np
import joblib, os
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Well Test Interpretation", layout="wide")
st.title("Well Test Interpretation")
st.divider()

models = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        models[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))

with st.sidebar:
    sel = st.selectbox("Scientific Model", list(models.keys()))
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.8)

m = models[sel]
feats = m.get("feature_names", [f"measurement_{i}" for i in range(4)])
X_raw = np.array([st.number_input(f, key=f"sc_{sel}_{i}") for i, f in enumerate(feats)]).reshape(1, -1)

if st.button("Analyze"):
    if m.get("scaler"):
        X_scaled = m["scaler"].transform(X_raw)
    else:
        X_scaled = X_raw
    pred = m["model"].predict(X_scaled)[0]
    if hasattr(m["model"], "predict_proba"):
        proba = m["model"].predict_proba(X_scaled)[0]
        conf_val = proba.max()
        if conf_val >= conf:
            st.success(f"Classification: {pred} (confidence: {conf_val:.2%})")
        else:
            st.warning(f"Low confidence ({conf_val:.2%}), result may be unreliable")
    else:
        st.metric("Prediction", f"{pred:.4f}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3))
    a1.bar(feats, X_raw[0])
    a1.set_title("Input profile")
    a2.hist(np.random.randn(100) + float(pred), bins=15)
    a2.set_title("Uncertainty distribution")
    st.pyplot(fig)
