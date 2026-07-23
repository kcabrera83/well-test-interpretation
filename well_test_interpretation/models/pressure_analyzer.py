"""Gradient Boosting model for pressure derivative analysis and flow regime identification."""

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


FLOW_REGIME_NAMES = {
    0: "Wellbore Storage",
    1: "Radial Flow",
    2: "Boundary Dominated",
}


class PressureAnalyzer:
    """GradientBoosting classifier for identifying flow regimes from pressure derivative data.

    Attributes
    ----------
    model : GradientBoostingClassifier
        The trained model.
    feature_names : list[str]
        Names of input features.
    trained : bool
        Whether the model has been trained.
    """

    FLOW_REGIMES = {0: "Wellbore Storage", 1: "Radial Flow", 2: "Boundary Dominated"}

    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42):
        """Initialize the PressureAnalyzer.

        Parameters
        ----------
        n_estimators : int
            Number of boosting stages.
        max_depth : int
            Maximum depth of individual regression estimators.
        learning_rate : float
            Learning rate shrinks the contribution of each tree.
        random_state : int
            Random state for reproducibility.
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=random_state,
        )
        self.feature_names = []
        self.trained = False

    def train(self, X_train, y_train, feature_names=None):
        """Train the flow regime classifier.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        feature_names : list[str], optional
            Feature names for importance reporting.

        Returns
        -------
        dict
            Training metrics.
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self.model.fit(X_train, y_train)
        self.trained = True

        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        return {"train_accuracy": train_acc, "n_samples": len(y_train)}

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            Test labels.

        Returns
        -------
        dict
            Evaluation metrics including accuracy, classification report,
            confusion matrix, and feature importances.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=[FLOW_REGIME_NAMES.get(i, f"Class {i}") for i in sorted(set(y_test))],
            output_dict=True,
        )
        cm = confusion_matrix(y_test, y_pred)
        importances = self.model.feature_importances_

        feature_importance = dict(zip(self.feature_names, importances.tolist()))

        return {
            "test_accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "feature_importances": feature_importance,
        }

    def predict(self, X):
        """Predict flow regimes.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted flow regime labels.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict flow regime probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted probabilities per class.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict_proba(X)

    def save(self, path):
        """Save the model to disk.

        Parameters
        ----------
        path : str
            File path for the saved model.
        """
        joblib.dump({"model": self.model, "features": self.feature_names}, path)

    @classmethod
    def load(cls, path):
        """Load a model from disk.

        Parameters
        ----------
        path : str
            File path of the saved model.

        Returns
        -------
        PressureAnalyzer
            Loaded analyzer instance.
        """
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["features"]
        instance.trained = True
        return instance
