"""lmfit-based model for estimating permeability and skin factor from pressure data."""

import numpy as np
import joblib
from lmfit import Model
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")


class ReservoirEstimator:
    """lmfit-based estimator for permeability and skin factor from pressure transient data."""

    def __init__(self, **kwargs):
        self.feature_names = []
        self.trained = False
        self._perm_params = None
        self._skin_params = None
        self._perm_stats = None
        self._skin_stats = None

    @staticmethod
    def _perm_model(X, a, b, c, d, e):
        t = X[:, 0] if X.ndim > 1 else X
        p = X[:, 1] if X.ndim > 1 else np.ones_like(t)
        q = X[:, 2] if X.ndim > 1 else np.ones_like(t)
        return a * np.log10(np.maximum(t, 1e-10)) + b * p / (q + 1e-10) + c * t + d + e * np.sqrt(np.maximum(t, 1e-10))

    @staticmethod
    def _skin_model(X, a, b, c, d, e):
        t = X[:, 0] if X.ndim > 1 else X
        p = X[:, 1] if X.ndim > 1 else np.ones_like(t)
        rw = X[:, 3] if X.ndim > 1 and X.shape[1] > 3 else np.ones_like(t)
        return a * np.log10(np.maximum(rw, 1e-10)) + b * np.log10(np.maximum(t, 1e-10)) + c * p + d + e * rw

    def train(self, X_train, y_perm_train, y_skin_train, feature_names=None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self.trained = True

        lmfit_perm = Model(self._perm_model)
        self._perm_params = lmfit_perm.make_params(
            a=100.0, b=0.5, c=0.01, d=10.0, e=1.0
        )
        try:
            perm_result = lmfit_perm.fit(y_perm_train, self._perm_params, X=X_train)
            self._perm_params = perm_result.params
            self._perm_stats = {
                "r_squared": 1 - (perm_result.residual.var() / np.var(y_perm_train)),
                "rmse": np.sqrt(np.mean(perm_result.residual ** 2)),
            }
        except Exception:
            popt, _ = curve_fit(self._perm_model, X_train, y_perm_train, maxfev=10000)
            self._perm_params = popt
            pred = self._perm_model(X_train, *popt)
            self._perm_stats = {
                "r_squared": 1 - np.var(y_perm_train - pred) / np.var(y_perm_train),
                "rmse": np.sqrt(np.mean((y_perm_train - pred) ** 2)),
            }

        lmfit_skin = Model(self._skin_model)
        self._skin_params = lmfit_skin.make_params(
            a=1.0, b=0.5, c=0.001, d=0.0, e=0.1
        )
        try:
            skin_result = lmfit_skin.fit(y_skin_train, self._skin_params, X=X_train)
            self._skin_params = skin_result.params
            self._skin_stats = {
                "r_squared": 1 - (skin_result.residual.var() / np.var(y_skin_train)),
                "rmse": np.sqrt(np.mean(skin_result.residual ** 2)),
            }
        except Exception:
            popt, _ = curve_fit(self._skin_model, X_train, y_skin_train, maxfev=10000)
            self._skin_params = popt
            pred = self._skin_model(X_train, *popt)
            self._skin_stats = {
                "r_squared": 1 - np.var(y_skin_train - pred) / np.var(y_skin_train),
                "rmse": np.sqrt(np.mean((y_skin_train - pred) ** 2)),
            }

        perm_pred = self._predict_perm_raw(X_train)
        skin_pred = self._predict_skin_raw(X_train)

        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        return {
            "perm_train_r2": r2_score(y_perm_train, perm_pred),
            "perm_train_rmse": np.sqrt(mean_squared_error(y_perm_train, perm_pred)),
            "skin_train_r2": r2_score(y_skin_train, skin_pred),
            "skin_train_rmse": np.sqrt(mean_squared_error(y_skin_train, skin_pred)),
        }

    def _predict_perm_raw(self, X):
        if hasattr(self._perm_params, 'values'):
            vals = [self._perm_params[k].value for k in sorted(self._perm_params.keys())]
        else:
            vals = self._perm_params
        return self._perm_model(X, *vals)

    def _predict_skin_raw(self, X):
        if hasattr(self._skin_params, 'values'):
            vals = [self._skin_params[k].value for k in sorted(self._skin_params.keys())]
        else:
            vals = self._skin_params
        return self._skin_model(X, *vals)

    def evaluate(self, X_test, y_perm_test, y_skin_test):
        if not self.trained:
            raise RuntimeError("Models must be trained before evaluation.")

        perm_pred = self._predict_perm_raw(X_test)
        skin_pred = self._predict_skin_raw(X_test)

        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        return {
            "permeability": {
                "r2": r2_score(y_perm_test, perm_pred),
                "rmse": np.sqrt(mean_squared_error(y_perm_test, perm_pred)),
                "mae": mean_absolute_error(y_perm_test, perm_pred),
            },
            "skin_factor": {
                "r2": r2_score(y_skin_test, skin_pred),
                "rmse": np.sqrt(mean_squared_error(y_skin_test, skin_pred)),
                "mae": mean_absolute_error(y_skin_test, skin_pred),
            },
            "feature_importances": self._estimate_feature_importance(X_test, y_perm_test),
        }

    def _estimate_feature_importance(self, X, y):
        base_rmse = np.sqrt(np.mean((y - self._predict_perm_raw(X)) ** 2))
        importances = []
        for i in range(X.shape[1]):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            perm_rmse = np.sqrt(np.mean((y - self._predict_perm_raw(X_perm)) ** 2))
            importances.append(max(perm_rmse - base_rmse, 0))
        importances = np.array(importances)
        total = importances.sum()
        if total > 0:
            importances /= total
        else:
            importances = np.ones(X.shape[1]) / X.shape[1]
        return dict(zip(self.feature_names, importances.tolist()))

    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Models must be trained before prediction.")
        return self._predict_perm_raw(X), self._predict_skin_raw(X)

    def feature_importances(self):
        return np.ones(len(self.feature_names)) / len(self.feature_names) if self.feature_names else np.array([])

    def save(self, path):
        data = {
            "feature_names": self.feature_names,
            "trained": self.trained,
            "perm_params": self._perm_params,
            "skin_params": self._skin_params,
            "perm_stats": self._perm_stats,
            "skin_stats": self._skin_stats,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.feature_names = data["feature_names"]
        instance.trained = data["trained"]
        instance._perm_params = data["perm_params"]
        instance._skin_params = data["skin_params"]
        instance._perm_stats = data.get("perm_stats")
        instance._skin_stats = data.get("skin_stats")
        return instance
