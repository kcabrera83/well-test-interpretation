"""Generate synthetic pressure transient test data for well testing analysis."""

import numpy as np
import pandas as pd


def generate_pressure_transient_data(n_samples=5000, seed=42):
    """Generate synthetic well test data with realistic physics-based relationships.

    Parameters
    ----------
    n_samples : int
        Number of data points to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing time_hours, pressure_psi, flow_rate_bbl_d,
        permeability_md, skin_factor, wellbore_radius_ft, reservoir_pressure_psi,
        drainage_radius_ft, formation_thickness_ft.
    """
    rng = np.random.default_rng(seed)

    permeability_md = rng.uniform(1.0, 500.0, n_samples)
    skin_factor = rng.uniform(-2.0, 10.0, n_samples)
    wellbore_radius_ft = rng.uniform(0.25, 0.5, n_samples)
    reservoir_pressure_psi = rng.uniform(2000.0, 6000.0, n_samples)
    drainage_radius_ft = rng.uniform(500.0, 3000.0, n_samples)
    formation_thickness_ft = rng.uniform(10.0, 200.0, n_samples)

    flow_rate_bbl_d = rng.uniform(50.0, 2000.0, n_samples)

    time_hours = rng.exponential(scale=24.0, size=n_samples)
    time_hours = np.sort(time_hours)
    time_hours = np.clip(time_hours, 0.001, 500.0)

    mu = 1.0
    phi = 0.2
    ct = 1e-5
    bo = 1.2

    log_time = np.log10(time_hours + 1e-6)

    pressure_drop = (
        70.6 * flow_rate_bbl_d * mu * bo / (permeability_md * formation_thickness_ft)
    ) * (
        np.log(4.0 * permeability_md * time_hours / (phi * mu * ct * wellbore_radius_ft**2)) - 0.89
        + 2.0 * skin_factor / np.log(10)
    )

    pressure_psi = reservoir_pressure_psi - np.abs(pressure_drop)

    pressure_psi += rng.normal(0, 5.0, n_samples)

    flow_regime = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        t = time_hours[i]
        k = permeability_md[i]
        r = drainage_radius_ft[i]
        t_boundary = (phi * mu * ct * r**2) / (0.00632 * k)
        if t < t_boundary * 0.01:
            flow_regime[i] = 0
        elif t < t_boundary * 0.5:
            flow_regime[i] = 1
        else:
            flow_regime[i] = 2

    df = pd.DataFrame({
        "time_hours": time_hours,
        "pressure_psi": pressure_psi,
        "flow_rate_bbl_d": flow_rate_bbl_d,
        "permeability_md": permeability_md,
        "skin_factor": skin_factor,
        "wellbore_radius_ft": wellbore_radius_ft,
        "reservoir_pressure_psi": reservoir_pressure_psi,
        "drainage_radius_ft": drainage_radius_ft,
        "formation_thickness_ft": formation_thickness_ft,
        "flow_regime": flow_regime,
        "log_time": log_time,
    })

    return df


def generate_batch_data(n_sets=10, samples_per_set=500, seed=42):
    """Generate multiple well test datasets.

    Parameters
    ----------
    n_sets : int
        Number of well test sets.
    samples_per_set : int
        Samples per well test.
    seed : int
        Random seed.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames, one per well test.
    """
    datasets = []
    for i in range(n_sets):
        df = generate_pressure_transient_data(
            n_samples=samples_per_set, seed=seed + i
        )
        df["well_id"] = i
        datasets.append(df)
    return datasets
