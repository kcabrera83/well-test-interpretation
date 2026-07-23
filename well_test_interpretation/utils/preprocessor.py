"""Preprocessing utilities for well test interpretation."""

import numpy as np
import pandas as pd


def compute_pressure_derivative(df):
    """Compute Bourdet pressure derivative."""
    df = df.sort_values("time_hours").copy()
    log_t = np.log10(df["time_hours"].values)
    dp = np.gradient(df["pressure_psi"].values, log_t)
    df["dp_dlogt"] = dp
    return df


def extract_features(df):
    """Extract features for flow regime identification."""
    df = compute_pressure_derivative(df)

    df["log_time"] = np.log10(df["time_hours"].values + 1e-6)
    df["pressure_squared"] = df["pressure_psi"].values ** 2
    df["normalized_pressure"] = (
        df["pressure_psi"].values / df["reservoir_pressure_psi"].values
    )
    df["flow_normalized_dp"] = (
        df["dp_dlogt"].values / (df["flow_rate_bbl_d"].values + 1e-6)
    )
    df["kh_ratio"] = (
        df["permeability_md"].values * df["formation_thickness_ft"].values
    )
    df["well_index"] = (
        df["permeability_md"].values * df["formation_thickness_ft"].values
        / (141.2 * 1.0 * 1.2 * (np.log(df["drainage_radius_ft"].values / df["wellbore_radius_ft"].values) - 0.75 + 2.0 * df["skin_factor"].values))
    )

    return df


def _standardize(X_train, X_test=None):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-10
    X_train_s = (X_train - mean) / std
    if X_test is not None:
        X_test_s = (X_test - mean) / std
        return X_train_s, X_test_s, mean, std
    return X_train_s, mean, std


def prepare_flow_regime_data(df, target_col="flow_regime"):
    """Prepare features and target for flow regime classification."""
    feature_cols = [
        "log_time", "pressure_psi", "dp_dlogt", "flow_rate_bbl_d",
        "normalized_pressure", "flow_normalized_dp", "pressure_squared",
        "wellbore_radius_ft", "formation_thickness_ft",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df[target_col].values

    n = len(X)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(n * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train, X_test, mean, std = _standardize(X_train, X_test)

    return X_train, X_test, y_train, y_test, available, {"mean": mean, "std": std}


def prepare_reservoir_data(df, target_col="permeability_md"):
    """Prepare features and target for reservoir parameter estimation."""
    feature_cols = [
        "time_hours", "pressure_psi", "flow_rate_bbl_d",
        "wellbore_radius_ft", "reservoir_pressure_psi",
        "drainage_radius_ft", "formation_thickness_ft",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df[target_col].values

    n = len(X)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(n * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train, X_test, mean, std = _standardize(X_train, X_test)

    return X_train, X_test, y_train, y_test, available, {"mean": mean, "std": std}
