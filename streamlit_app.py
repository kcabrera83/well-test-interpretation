import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Well Test PTA", layout="wide")
st.title("Well Test PTA")

p = Path(__file__).parent / 'outputs' / 'models'
models = {'regime': joblib.load(p / 'flow_regime_classifier.pkl'), 'perm': joblib.load(p / 'permeability_estimator.pkl'), 'skin': joblib.load(p / 'skin_estimator.pkl')}

tab1, tab2, tab3 = st.tabs(['Predict', 'Charts', 'Info'])

with tab1:
    st.subheader('Inputs')
    c = st.columns(2)
    time = c[0].slider('Time', 0, 1000, 500)
    pres = c[1].slider('Pres', 1000, 10000, 5500)
    deriv = c[0].slider('Deriv', 0, 1000, 500)
    rate = c[1].slider('Rate', 100, 10000, 5050)
    if st.button('Run', type='primary'):
        x = np.array([[time, pres, deriv, rate]])
        res = {}
        m = models['regime']
        if isinstance(m, dict):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            res['regime'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
        else:
            res['regime'] = float(m.predict(x)[0])
        m = models['perm']
        if isinstance(m, dict):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            res['perm'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
        else:
            res['perm'] = float(m.predict(x)[0])
        m = models['skin']
        if isinstance(m, dict):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            res['skin'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
        else:
            res['skin'] = float(m.predict(x)[0])
        st.divider()
        rc = st.columns(len(res))
        for i, (k, v) in enumerate(res.items()):
            rc[i].metric(k.replace('_',' ').title(), str(v) if isinstance(v,str) else f'{v:.2f}')

with tab2:
    st.info('Charts update after prediction')

with tab3:
    st.markdown('Pressure transient analysis with flow regime identification')
    st.caption('Built with scikit-learn + Streamlit')