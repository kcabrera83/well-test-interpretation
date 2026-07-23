"""Curve fitting model for pressure derivative analysis and flow regime identification."""

import numpy as np
import joblib
from scipy.optimize import curve_fit
from lmfit import Model
import warnings

warnings.filterwarnings("ignore")

FLOW_REGIME_NAMES = {
    0: "Wellbore Storage",
    1: "Radial Flow",
    2: "Boundary Dominated",
}


class PressureAnalyzer:
    """Curve-fitting based analyzer for identifying flow regimes from pressure derivative data."""

    FLOW_REGIMES = {0: "Wellbore Storage", 1: "Radial Flow", 2: "Boundary Dominated"}

    def __init__(self, **kwargs):
        self.feature_names = []
        self.trained = False
        self._fit_results = {}
        self._scaler_mean = None
        self._scaler_std = None

    @staticmethod
    def _pressure_model(t, pi, slope, intercept, noise_amp):
        log_t = np.log10(np.maximum(t, 1e-10))
        return pi + slope * log_t + intercept

    @staticmethod
    def _radial_flow_model(t, k_norm, dp_radial, p_mid):
        return p_mid + dp_radial * np.log10(np.maximum(t, 1e-10)) * k_norm

    @staticmethod
    def _storage_model(t, c_d, p_start, rate):
        return p_start + rate * np.log10(np.maximum(t, 1e-10)) / (c_d + 1e-10)

    def _classify_flow_regime(self, t, p):
        if len(t) < 3:
            return 1
        log_t = np.log10(np.maximum(t, 1e-10))
        try:
            coeffs = np.polyfit(log_t, p, 2)
            curvature = coeffs[0]
        except Exception:
            curvature = 0

        dp = np.diff(p)
        d2p = np.diff(dp) if len(dp) > 1 else np.array([0])

        if abs(curvature) > 10 and np.std(dp[:max(3, len(dp)//5)]) > np.std(dp[-max(3, len(dp)//5):]):
            return 0
        elif abs(curvature) < 2 and len(dp) > 5:
            mid = len(dp) // 2
            if np.std(dp[mid:]) < 0.5 * (np.std(dp[:mid]) + 1e-10):
                return 1
            else:
                return 2
        else:
            if abs(curvature) > 5:
                return 2
            return 1

    def train(self, X_train, y_train, feature_names=None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self.trained = True

        unique_classes = np.unique(y_train)
        self._class_counts = {}
        for c in unique_classes:
            self._class_counts[int(c)] = int(np.sum(y_train == c))

        total = len(y_train)
        self._prior_probs = {int(c): cnt / total for c, cnt in self._class_counts.items()}

        self._regime_profiles = {}
        for c in unique_classes:
            mask = y_train == c
            X_c = X_train[mask]
            if len(X_c) > 0:
                self._regime_profiles[int(c)] = {
                    "mean": X_c.mean(axis=0).tolist(),
                    "std": X_c.std(axis=0).tolist(),
                }

        return {"train_accuracy": self._score_internal(X_train, y_train), "n_samples": len(y_train)}

    def _score_internal(self, X, y):
        preds = [self._predict_single(x) for x in X]
        return np.mean(np.array(preds) == y)

    def _predict_single(self, x):
        best_regime = 0
        best_score = -1
        for regime, profile in self._regime_profiles.items():
            mean = np.array(profile["mean"])
            std = np.array(profile["std"]) + 1e-10
            z = -0.5 * np.sum(((x - mean) / std) ** 2)
            score = z + np.log(self._prior_probs.get(regime, 1e-10))
            if score > best_score:
                best_score = score
                best_regime = regime
        return best_regime

    def evaluate(self, X_test, y_test):
        if not self.trained:
            raise RuntimeError("Model must be trained before evaluation.")

        preds = np.array([self._predict_single(x) for x in X_test])
        acc = np.mean(preds == y_test)

        importances = np.zeros(len(self.feature_names))
        for i in range(len(self.feature_names)):
            perm = X_test.copy()
            np.random.shuffle(perm[:, i])
            perm_preds = np.array([self._predict_single(x) for x in perm])
            importances[i] = acc - np.mean(perm_preds == y_test)
        importances = np.maximum(importances, 0)
        total = importances.sum()
        if total > 0:
            importances /= total
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        return {
            "test_accuracy": acc,
            "feature_importances": dict(zip(self.feature_names, importances.tolist())),
        }

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        probas = []
        for x in X:
            scores = []
            for regime in sorted(self._regime_profiles.keys()):
                mean = np.array(self._regime_profiles[regime]["mean"])
                std = np.array(self._regime_profiles[regime]["std"]) + 1e-10
                z = -0.5 * np.sum(((x - mean) / std) ** 2)
                scores.append(z + np.log(self._prior_probs.get(regime, 1e-10)))
            scores = np.array(scores)
            scores -= scores.max()
            exp_scores = np.exp(scores)
            probas.append(exp_scores / exp_scores.sum())
        return np.array(probas)

    def fit_curve_with_lmfit(self, t, p):
        model = Model(self._pressure_model)
        params = model.make_params(pi=p.max(), slope=-500, intercept=0, noise_amp=1)
        params["slope"].min = -10000
        params["slope"].max = 0
        try:
            result = model.fit(p, params, t=t)
            return result
        except Exception:
            return None

    def save(self, path):
        joblib.dump({
            "feature_names": self.feature_names,
            "trained": self.trained,
            "regime_profiles": self._regime_profiles,
            "prior_probs": self._prior_probs,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.feature_names = data["feature_names"]
        instance.trained = data["trained"]
        instance._regime_profiles = data["regime_profiles"]
        instance._prior_probs = data["prior_probs"]
        return instance
