"""Preprocessing utilities for well test interpretation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def compute_pressure_derivative(df):
    """Compute Bourdet pressure derivative.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'time_hours' and 'pressure_psi' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'dp_dlogt' column.
    """
    df = df.sort_values("time_hours").copy()
    log_t = np.log10(df["time_hours"].values)
    dp = np.gradient(df["pressure_psi"].values, log_t)
    df["dp_dlogt"] = dp
    return df


def extract_features(df):
    """Extract features for flow regime identification.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'time_hours', 'pressure_psi', 'flow_rate_bbl_d',
        'permeability_md', 'skin_factor', 'wellbore_radius_ft',
        'reservoir_pressure_psi', 'drainage_radius_ft',
        'formation_thickness_ft'.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features.
    """
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


def prepare_flow_regime_data(df, target_col="flow_regime"):
    """Prepare features and target for flow regime classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and flow_regime target.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names, scaler.
    """
    feature_cols = [
        "log_time", "pressure_psi", "dp_dlogt", "flow_rate_bbl_d",
        "normalized_pressure", "flow_normalized_dp", "pressure_squared",
        "wellbore_radius_ft", "formation_thickness_ft",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, available, scaler


def prepare_reservoir_data(df, target_col="permeability_md"):
    """Prepare features and target for reservoir parameter estimation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names, scaler.
    """
    feature_cols = [
        "time_hours", "pressure_psi", "flow_rate_bbl_d",
        "wellbore_radius_ft", "reservoir_pressure_psi",
        "drainage_radius_ft", "formation_thickness_ft",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, available, scaler
