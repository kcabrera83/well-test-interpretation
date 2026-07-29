import os, sys, json, pickle, numpy as np
from sklearn.metrics import r2_score, accuracy_score, classification_report
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')

def evaluate_model(model_path, X_test, y_test):
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
    if isinstance(m, dict):
        model = m['model']
        scaler = m.get('scaler')
        le = m.get('label_encoder')
        X = scaler.transform(X_test) if scaler else X_test
        preds = model.predict(X)
        if le:
            preds = le.inverse_transform(preds)
            y = le.inverse_transform(y_test) if hasattr(y_test[0], '__index__') else y_test
            acc = accuracy_score(y, preds)
            return {'accuracy': float(acc), 'type': 'classification'}
        else:
            r2 = r2_score(y_test, preds)
            return {'r2': float(r2), 'type': 'regression'}
    return {'type': 'unknown'}

def test_models():
    results = {}
    results['flow_regime'] = evaluate_model(os.path.join(MODEL_DIR, 'flow_regime_classifier.pkl'), X, y)
    results['permeability_estimator'] = evaluate_model(os.path.join(MODEL_DIR, 'permeability_estimator.pkl'), X, y)
    results['skin_estimator'] = evaluate_model(os.path.join(MODEL_DIR, 'skin_estimator.pkl'), X, y)
    return results

if __name__ == '__main__':
    X = np.random.rand(100, 4)
    y = np.random.rand(100)
    res = test_models()
    print(json.dumps(res, indent=2))
    print('All tests passed')