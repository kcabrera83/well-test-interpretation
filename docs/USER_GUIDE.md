# User Guide - Well Test Interpretation

## Overview

The Well Test Interpretation system uses machine learning to analyze well test pressure data, identify flow regimes, and estimate reservoir parameters such as permeability and skin factor.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd well-test-interpretation
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

This generates 5,000 synthetic pressure transient records and trains:
- Pressure Analyzer (GradientBoostingClassifier) - Identifies flow regimes
- Reservoir Estimator (RandomForestRegressor) - Estimates permeability and skin

### Run the Server

```bash
python app.py
```

Open `http://localhost:5012` in your browser.

## Dashboard Features

- **Flow Regime Analysis Panel** - Upload pressure/time/flow data to identify flow regimes
- **Reservoir Estimation Panel** - Estimate permeability (md) and skin factor
- **Results Visualization** - Charts showing predictions across time series
- **Model Information** - View loaded models and classification labels

## API Usage

### Analyze Flow Regimes (Python)

```python
import requests

response = requests.post("http://localhost:5012/api/analyze", json={
    "time_hours": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    "pressure_psi": [4500, 4400, 4350, 4200, 4100, 3900, 3800],
    "flow_rate_bbl_d": [500, 500, 500, 500, 500, 500, 500]
})
result = response.json()
for regime, prob in zip(result['predictions'], result['probabilities']):
    print(f"Regime: {regime} (confidence: {max(prob):.2%})")
```

### Estimate Reservoir Parameters (Python)

```python
import requests

response = requests.post("http://localhost:5012/api/estimate", json={
    "time_hours": [0.1, 0.5, 1.0, 5.0, 10.0],
    "pressure_psi": [4500, 4400, 4350, 4200, 4100],
    "flow_rate_bbl_d": [500, 500, 500, 500, 500],
    "wellbore_radius_ft": [0.35, 0.35, 0.35, 0.35, 0.35],
    "reservoir_pressure_psi": [5000, 5000, 5000, 5000, 5000],
    "drainage_radius_ft": [1500, 1500, 1500, 1500, 1500],
    "formation_thickness_ft": [50, 50, 50, 50, 50]
})
result = response.json()
print(f"Permeability: {sum(result['permeability_md'])/len(result['permeability_md']):.1f} md")
print(f"Skin factor: {sum(result['skin_factor'])/len(result['skin_factor']):.2f}")
```

### Check Health

```bash
curl http://localhost:5012/api/health
```

### Get Model Info

```bash
curl http://localhost:5012/api/models
```

## Understanding Flow Regimes

| Regime | Time Period | Description |
|--------|-------------|-------------|
| Wellbore Storage | Early | Dominated by wellbore fluid compression/expansion |
| Radial Flow | Middle | Infinite-acting radial flow in reservoir |
| Boundary Dominated | Late | Feel reservoir boundaries, pseudo-steady state |

## Typical Workflow

1. Collect pressure transient test data (time, pressure, flow rate)
2. Provide well/reservoir parameters (radius, thickness, pressure)
3. Call `/api/analyze` to identify flow regimes across the test duration
4. Call `/api/estimate` to get permeability and skin factor
5. Use results for reservoir characterization

## Running Tests

```bash
python test_api.py
```

## Troubleshooting

- **Model not loaded**: Run `python train.py` first
- **Empty data arrays**: Ensure all arrays are non-empty and same length
- **503 error**: Models not trained yet - run train.py

---

*Elaborado por Ing. Kelvin Cabrera*
