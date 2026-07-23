"""Random Forest model for estimating permeability and skin factor from pressure data."""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ReservoirEstimator:
    """RandomForest regressor for estimating permeability and skin factor.

    Attributes
    ----------
    permeability_model : RandomForestRegressor
        Model for permeability estimation.
    skin_model : RandomForestRegressor
        Model for skin factor estimation.
    feature_names : list[str]
        Names of input features.
    trained : bool
        Whether the model has been trained.
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
    ):
        """Initialize the ReservoirEstimator.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.
        max_depth : int
            Maximum depth of each tree.
        min_samples_split : int
            Minimum samples required to split a node.
        random_state : int
            Random state for reproducibility.
        """
        self.permeability_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        self.skin_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names = []
        self.trained = False

    def train(self, X_train, y_perm_train, y_skin_train, feature_names=None):
        """Train both permeability and skin factor models.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_perm_train : np.ndarray
            Training permeability targets (md).
        y_skin_train : np.ndarray
            Training skin factor targets.
        feature_names : list[str], optional
            Feature names for importance reporting.

        Returns
        -------
        dict
            Training metrics.
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self.permeability_model.fit(X_train, y_perm_train)
        self.skin_model.fit(X_train, y_skin_train)
        self.trained = True

        perm_train_pred = self.permeability_model.predict(X_train)
        skin_train_pred = self.skin_model.predict(X_train)

        return {
            "perm_train_r2": r2_score(y_perm_train, perm_train_pred),
            "perm_train_rmse": np.sqrt(mean_squared_error(y_perm_train, perm_train_pred)),
            "skin_train_r2": r2_score(y_skin_train, skin_train_pred),
            "skin_train_rmse": np.sqrt(mean_squared_error(y_skin_train, skin_train_pred)),
        }

    def evaluate(self, X_test, y_perm_test, y_skin_test):
        """Evaluate both models on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        y_perm_test : np.ndarray
            True permeability values.
        y_skin_test : np.ndarray
            True skin factor values.

        Returns
        -------
        dict
            Evaluation metrics including R2, RMSE, MAE, and feature importances.
        """
        if not self.trained:
            raise RuntimeError("Models must be trained before evaluation.")

        perm_pred = self.permeability_model.predict(X_test)
        skin_pred = self.skin_model.predict(X_test)

        perm_r2 = r2_score(y_perm_test, perm_pred)
        perm_rmse = np.sqrt(mean_squared_error(y_perm_test, perm_pred))
        perm_mae = mean_absolute_error(y_perm_test, perm_pred)

        skin_r2 = r2_score(y_skin_test, skin_pred)
        skin_rmse = np.sqrt(mean_squared_error(y_skin_test, skin_pred))
        skin_mae = mean_absolute_error(y_skin_test, skin_pred)

        return {
            "permeability": {"r2": perm_r2, "rmse": perm_rmse, "mae": perm_mae},
            "skin_factor": {"r2": skin_r2, "rmse": skin_rmse, "mae": skin_mae},
            "feature_importances": dict(
                zip(self.feature_names, self.feature_importances().tolist())
            ),
        }

    def predict(self, X):
        """Predict permeability and skin factor.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Predicted permeability (md) and skin factor arrays.
        """
        if not self.trained:
            raise RuntimeError("Models must be trained before prediction.")
        return self.permeability_model.predict(X), self.skin_model.predict(X)

    def feature_importances(self):
        """Get combined feature importances (average of both models).

        Returns
        -------
        np.ndarray
            Average feature importances.
        """
        return (
            self.permeability_model.feature_importances_
            + self.skin_model.feature_importances_
        ) / 2.0

    def save(self, path):
        """Save both models to disk.

        Parameters
        ----------
        path : str
            File path for the saved models.
        """
        joblib.dump(
            {
                "perm_model": self.permeability_model,
                "skin_model": self.skin_model,
                "features": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        """Load models from disk.

        Parameters
        ----------
        path : str
            File path of the saved models.

        Returns
        -------
        ReservoirEstimator
            Loaded estimator instance.
        """
        data = joblib.load(path)
        instance = cls()
        instance.permeability_model = data["perm_model"]
        instance.skin_model = data["skin_model"]
        instance.feature_names = data["features"]
        instance.trained = True
        return instance
